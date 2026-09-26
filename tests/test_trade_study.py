import math

from orbitforge.analysis import trade_study, monte_carlo
from orbitforge.core.errors import ReproducibilityError, ValidationError

G0 = 9.80665

def delta_v(m0, mf, isp):
    return isp * G0 * math.log(m0 / mf)

METRICS = {'delta_v': delta_v}

def scenario(name, **overrides):
    base = {
        'name': name,
        'metric': 'delta_v',
        'n': 256,
        'seed': 20260926,
        'distributions': {
            'm0': {'name': 'uniform', 'version': 1, 'params': {'low': 1000.0, 'high': 1100.0}},
            'mf': {'name': 'uniform', 'version': 1, 'params': {'low': 800.0, 'high': 900.0}},
            'isp': {'name': 'normal', 'version': 1, 'params': {'mu': 320.0, 'sigma': 5.0}},
        },
    }
    base.update(overrides)
    return base

def expect_reproducibility_error(fn):
    try:
        fn()
    except ReproducibilityError:
        return
    raise AssertionError('expected ReproducibilityError')

def test_same_manifest_runs_sample_identical():
    manifest = trade_study.build_manifest(
        'burn-uncertainty',
        [scenario('baseline'), scenario('reserve', seed=7)],
        quantiles=[0.05, 0.5, 0.95],
    )
    first = trade_study.run_manifest(manifest, METRICS, software_version='1.0.0')
    second = trade_study.run_manifest(manifest, METRICS, software_version='1.0.0')
    assert first == second
    for a, b in zip(first['scenarios'], second['scenarios']):
        assert a['samples_sha256'] == b['samples_sha256']
        assert a['quantiles'] == b['quantiles']
        assert set(a['quantiles']) == {'0.05', '0.5', '0.95'}

def test_per_sample_seeds_independent_of_order_and_dict_layout():
    s1 = scenario('a', seed=42)
    s2 = scenario('b', seed=99)
    m_forward = trade_study.build_manifest('order', [s1, s2])
    m_reversed = trade_study.build_manifest('order', [s2, s1])
    assert trade_study.manifest_fingerprint(m_forward) == trade_study.manifest_fingerprint(m_reversed)
    r_forward = trade_study.run_manifest(m_forward, METRICS)
    r_reversed = trade_study.run_manifest(m_reversed, METRICS)
    by_name_f = {s['name']: s for s in r_forward['scenarios']}
    by_name_r = {s['name']: s for s in r_reversed['scenarios']}
    assert by_name_f == by_name_r

def test_per_sample_draw_is_stable_when_n_grows():
    base = scenario('a', n=64)
    grown = scenario('a', n=128)
    small = monte_carlo.draw_samples(delta_v,
                                     {k: trade_study.sampler_for(spec) for k, spec in base['distributions'].items()},
                                     base['n'], base['seed'])
    large = monte_carlo.draw_samples(delta_v,
                                     {k: trade_study.sampler_for(spec) for k, spec in grown['distributions'].items()},
                                     grown['n'], grown['seed'])
    assert large[:64] == small

def test_config_changes_change_fingerprint_and_miss_cache():
    store = trade_study.ResultStore()
    m1 = trade_study.build_manifest('study', [scenario('a')])
    r1 = trade_study.run_cached(m1, METRICS, store, software_version='1.0.0')
    variants = {
        'seed': trade_study.build_manifest('study', [scenario('a', seed=43)]),
        'n': trade_study.build_manifest('study', [scenario('a', n=512)]),
        'quantiles': trade_study.build_manifest('study', [scenario('a')], quantiles=[0.05, 0.25, 0.95]),
        'study_name': trade_study.build_manifest('study2', [scenario('a')]),
        'dist_params': trade_study.build_manifest('study', [scenario('a', distributions={
            **scenario('a')['distributions'],
            'isp': {'name': 'normal', 'version': 1, 'params': {'mu': 320.0, 'sigma': 6.0}},
        })]),
    }
    fps = {r1['fingerprint']}
    for label, m in variants.items():
        r = trade_study.run_cached(m, METRICS, store, software_version='1.0.0')
        assert r['fingerprint'] not in fps, f'{label} collided with an existing result'
        fps.add(r['fingerprint'])
    assert len(store) == 1 + len(variants)

def test_distribution_version_change_is_a_distinct_config():
    trade_study.register_distribution('uniform', 9,
                                      lambda params: (lambda rng: rng.uniform(params['low'] - 1000.0, params['high'])))
    old = trade_study.build_manifest('v', [scenario('a')])
    new = trade_study.build_manifest('v', [scenario('a', distributions={
        'm0': {'name': 'uniform', 'version': 9, 'params': {'low': 1000.0, 'high': 1100.0}},
        'mf': scenario('a')['distributions']['mf'],
        'isp': scenario('a')['distributions']['isp'],
    })])
    assert trade_study.manifest_fingerprint(old) != trade_study.manifest_fingerprint(new)
    assert trade_study.run_manifest(old, METRICS) != trade_study.run_manifest(new, METRICS)

def test_software_version_participates_in_fingerprint():
    m = trade_study.build_manifest('study', [scenario('a')])
    assert trade_study.manifest_fingerprint(m, software_version='1.0.0') != \
           trade_study.manifest_fingerprint(m, software_version='1.0.1')

def test_verify_flags_tampered_or_wrong_results():
    m = trade_study.build_manifest('study', [scenario('a')])
    recorded = trade_study.run_manifest(m, METRICS, software_version='1.0.0')

    tampered_digest = {'fingerprint': recorded['fingerprint'], 'study': recorded['study'],
                       'scenarios': [dict(recorded['scenarios'][0], samples_sha256='0' * 64)]}
    expect_reproducibility_error(lambda: trade_study.verify_manifest(m, tampered_digest, METRICS, '1.0.0'))

    tampered_quantile = {'fingerprint': recorded['fingerprint'], 'study': recorded['study'],
                         'scenarios': [dict(recorded['scenarios'][0],
                                            quantiles={'0.05': 0.0, '0.5': 0.0, '0.95': 0.0})]}
    expect_reproducibility_error(lambda: trade_study.verify_manifest(m, tampered_quantile, METRICS, '1.0.0'))

    missing_scenario = {'fingerprint': recorded['fingerprint'], 'study': recorded['study'], 'scenarios': []}
    expect_reproducibility_error(lambda: trade_study.verify_manifest(m, missing_scenario, METRICS, '1.0.0'))

    m2 = trade_study.build_manifest('study', [scenario('a', seed=43)])
    expect_reproducibility_error(lambda: trade_study.verify_manifest(m2, recorded, METRICS, '1.0.0'))

    expect_reproducibility_error(lambda: trade_study.verify_manifest(m, recorded, METRICS, '1.0.1'))

def test_unknown_distribution_version_rejected_at_build():
    bad = scenario('a', distributions={
        **scenario('a')['distributions'],
        'isp': {'name': 'normal', 'version': 99, 'params': {'mu': 320.0, 'sigma': 5.0}}})
    try:
        trade_study.build_manifest('study', [bad])
    except ValidationError:
        pass
    else:
        raise AssertionError('expected ValidationError')

def test_run_cached_reuses_and_verifies_store_entry():
    store = trade_study.ResultStore()
    m = trade_study.build_manifest('study', [scenario('a')])
    first = trade_study.run_cached(m, METRICS, store, software_version='1.0.0')
    assert len(store) == 1
    second = trade_study.run_cached(m, METRICS, store, software_version='1.0.0')
    assert second is first
    first['scenarios'][0]['quantiles']['0.5'] += 1.0
    expect_reproducibility_error(lambda: trade_study.run_cached(m, METRICS, store, software_version='1.0.0'))
