import copy
import random

import pytest

from orbitforge.analysis import monte_carlo
from orbitforge.analysis.reproducible_trade import (
    CacheIntegrityError,
    ConfigurationError,
    EvaluationError,
    ManifestMismatchError,
    ReproducibleResultCache,
    SampleMismatchError,
    ScenarioDefinition,
    assert_reproducible,
    build_manifest,
    quantile,
    register_distribution,
    run_reproducible_trade,
    verify_result,
)

SCENARIOS = [
    {'id': 'leo', 'parameters': {'alt_km': 500}},
    {'id': 'meo', 'parameters': {'alt_km': 8000}},
]

# Construct with reversed insertion order on purpose: canonicalization must
# not depend on dict order.
DISTRIBUTIONS = {
    'mass_kg': {'type': 'uniform', 'low': 900.0, 'high': 1100.0},
    'isp_s': {'type': 'normal', 'mean': 300.0, 'stdev': 3.0},
}


def metric(inputs, parameters):
    return inputs['mass_kg'] * inputs['isp_s'] * 0.001 + parameters['alt_km']


def run_one(**overrides):
    kwargs = {
        'seed': 20260926,
        'samples': 64,
        'distribution_version': 'input-distributions-2026-09',
    }
    kwargs.update(overrides)
    return run_reproducible_trade(SCENARIOS, DISTRIBUTIONS, metric, **kwargs)


def test_repeated_run_is_sample_identical_and_keeps_trace():
    first = run_one()
    second = run_one()
    assert first == second
    assert assert_reproducible(first, second) is True
    verify_result(first)

    manifest = first['manifest']
    assert manifest['seed'] == 20260926
    assert manifest['samples_per_scenario'] == 64
    assert manifest['distribution_version'] == 'input-distributions-2026-09'
    assert manifest['quantiles'] == [0.05, 0.5, 0.95]
    assert manifest['scenario_ids'] == ['leo', 'meo']
    assert len(manifest['config_sha256']) == 64
    assert {k for k in manifest} == {
        'schema_version', 'seed', 'samples_per_scenario', 'quantiles', 'quantile_method',
        'distribution_version', 'scenario_ids', 'scenarios_sha256', 'distributions_sha256',
        'config_sha256',
    }

    leo = first['scenarios'][0]
    assert leo['scenario_id'] == 'leo'
    assert leo['sample_count'] == 64
    assert len(leo['samples']) == 64 and len(leo['values']) == 64
    assert set(leo['samples'][0]) == {'isp_s', 'mass_kg'}
    assert set(leo['quantiles']) == {'0.05', '0.5', '0.95'}


def test_scenario_and_distribution_order_do_not_change_samples():
    reversed_scenarios = list(reversed(SCENARIOS))
    reversed_distributions = {k: DISTRIBUTIONS[k] for k in reversed(DISTRIBUTIONS)}
    again = run_reproducible_trade(
        reversed_scenarios, reversed_distributions, metric,
        seed=20260926, samples=64, distribution_version='input-distributions-2026-09',
    )
    assert again == run_one()


def test_seed_change_gives_new_identity_without_cache_hit():
    first = run_one()
    cache = ReproducibleResultCache()
    first = run_one(cache=cache)
    other_seed = run_one(seed=7, cache=cache)

    assert len(cache) == 2
    assert other_seed['manifest']['config_sha256'] != first['manifest']['config_sha256']
    assert other_seed['manifest']['seed'] == 7
    assert other_seed['scenarios'][0]['samples'] != first['scenarios'][0]['samples']


def test_distribution_version_is_tracked_but_does_not_reseed_samples():
    first = run_one()
    second = run_one(distribution_version='input-distributions-2026-10')
    assert second['manifest']['distribution_version'] != first['manifest']['distribution_version']
    assert second['manifest']['config_sha256'] != first['manifest']['config_sha256']
    assert second['scenarios'][0]['samples'] == first['scenarios'][0]['samples']


def test_quantiles_are_recorded_in_manifest_and_result():
    result = run_one(quantiles=(0.9, 0.1, 0.5), quantile_method='nearest_rank')
    assert result['manifest']['quantiles'] == [0.1, 0.5, 0.9]
    assert result['manifest']['quantile_method'] == 'nearest_rank'
    assert list(result['scenarios'][0]['quantiles']) == ['0.1', '0.5', '0.9']
    assert result['manifest']['config_sha256'] != run_one()['manifest']['config_sha256']


def test_adding_scenarios_keeps_existing_scenario_samples_stable():
    leo_only = run_reproducible_trade(
        SCENARIOS[:1], DISTRIBUTIONS, metric,
        seed=20260926, samples=64, distribution_version='input-distributions-2026-09',
    )
    both = run_one()
    assert both['scenarios'][0]['samples'] == leo_only['scenarios'][0]['samples']


def test_cache_returns_verified_copy_for_identical_configuration():
    cache = ReproducibleResultCache()
    first = run_one(cache=cache)
    second = run_one(cache=cache)
    assert len(cache) == 1
    assert second == first
    assert second is not first
    assert second['scenarios'][0] is not first['scenarios'][0]


def test_cache_rejects_manifest_mismatch_and_tampering():
    first = run_one()
    other = run_one(seed=11)
    cache = ReproducibleResultCache()

    with pytest.raises(ManifestMismatchError):
        cache.put(other['manifest'], first)

    cache.put(first['manifest'], first)
    tampered = copy.deepcopy(first)
    tampered['scenarios'][0]['quantiles']['0.5'] += 1.0
    with pytest.raises(CacheIntegrityError):
        cache.put(first['manifest'], tampered)

    cache._records[first['manifest']['config_sha256']] = tampered
    with pytest.raises(CacheIntegrityError):
        run_one(cache=cache)


def test_sample_mismatch_is_reported_for_nondeterministic_evaluator():
    reference = run_one()

    def flaky(inputs, parameters):
        return metric(inputs, parameters) + random.random()

    with pytest.raises(SampleMismatchError) as exc_info:
        run_reproducible_trade(
            SCENARIOS, DISTRIBUTIONS, flaky, seed=20260926, samples=64,
            distribution_version='input-distributions-2026-09', reference=reference,
        )
    assert exc_info.value.mismatches[0]['scope'] in ('value', 'manifest', 'sample')


def test_invalid_configuration_raises_clear_errors():
    with pytest.raises(ConfigurationError):
        run_reproducible_trade(
            [{'id': 'a'}, {'id': 'a'}], DISTRIBUTIONS, metric,
            seed=1, samples=4, distribution_version='v1',
        )
    with pytest.raises(ConfigurationError):
        run_reproducible_trade(
            SCENARIOS, {'x': {'type': 'no-such-distribution'}}, metric,
            seed=1, samples=4, distribution_version='v1',
        )
    with pytest.raises(ConfigurationError):
        run_one(quantiles=(0.0, 0.5))
    with pytest.raises(ConfigurationError):
        run_one(quantiles=(0.5, 0.5))
    with pytest.raises(ConfigurationError):
        run_one(seed=None)
    with pytest.raises(ConfigurationError):
        run_one(samples=0)
    with pytest.raises(ConfigurationError):
        run_reproducible_trade(
            SCENARIOS, DISTRIBUTIONS, metric,
            seed=1, samples=4, distribution_version='',
        )


def test_non_finite_metric_fails_loudly():
    def bad(inputs, parameters):
        return float('nan')

    with pytest.raises(EvaluationError, match="scenario 'leo' sample 0"):
        run_reproducible_trade(
            SCENARIOS, {'x': {'type': 'constant', 'value': 1.0}}, bad,
            seed=1, samples=4, distribution_version='v1',
        )


def test_quantile_methods_are_deterministic():
    values = [1.0, 2.0, 3.0, 4.0]
    assert quantile(values, 0.5) == 2.5
    assert quantile(values, 0.5, method='nearest_rank') == 2.0
    with pytest.raises(ConfigurationError):
        quantile(values, 0.5, method='mystery')


def test_custom_and_builtin_distributions():
    register_distribution('test_offset_plus_one', lambda rng, spec: spec['base'] + 1)
    result = run_reproducible_trade(
        [ScenarioDefinition('s', parameters={'k': 2.0})],
        {
            'fixed': {'type': 'constant', 'value': 'ok'},
            'off': {'type': 'test_offset_plus_one', 'base': 41},
            'pick': {'type': 'choice', 'values': [1, 2, 3]},
            'n': {'type': 'randint', 'low': -2, 'high': 2},
            'tri': {'type': 'triangular', 'low': 0.0, 'mode': 1.0, 'high': 2.0},
        },
        lambda inputs, parameters: float(inputs['off']) * parameters['k'],
        seed='seed-string', samples=8, distribution_version='custom-v1',
    )
    rows = result['scenarios'][0]['samples']
    assert all(row['fixed'] == 'ok' and row['off'] == 42 for row in rows)
    assert all(row['pick'] in (1, 2, 3) and -2 <= row['n'] <= 2 for row in rows)
    assert all(0.0 <= row['tri'] <= 2.0 for row in rows)
    assert result['manifest']['seed'] == 'seed-string'
    assert all(row['n'] == row['n'] for row in rows)


def test_manifest_helpers_and_monte_carlo_reexports():
    manifest = build_manifest(
        SCENARIOS, DISTRIBUTIONS, seed=20260926, samples=64,
        distribution_version='input-distributions-2026-09',
    )
    assert manifest == run_one()['manifest']
    assert monte_carlo.run_trade is run_reproducible_trade
