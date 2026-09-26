"""Reproducible uncertainty sampling for trade studies.

This module turns one-shot Monte Carlo statistics into a traceable,
reproducible trade artifact:

* the scenario list, random seed, input-distribution version and reported
  quantiles are all recorded in a content-addressed manifest;
* every scenario keeps its per-sample inputs and metric values, so rerunning
  the same manifest must reproduce them sample by sample;
* results can be cached by configuration digest without cross-configuration
  hits, and any inconsistency raises an explicit error.

Determinism notes: each (scenario, sample index) pair draws from its own
``random.Random`` instance seeded by a SHA-256 derived key, so results do not
depend on scenario ordering, distribution insertion order, dict iteration
order or the host machine.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import random
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

SCHEMA_VERSION = 1
DEFAULT_QUANTILES = (0.05, 0.5, 0.95)
QUANTILE_METHODS = ('linear', 'nearest_rank')

__all__ = [
    'SCHEMA_VERSION',
    'DEFAULT_QUANTILES',
    'QUANTILE_METHODS',
    'ReproducibilityError',
    'ConfigurationError',
    'EvaluationError',
    'ManifestMismatchError',
    'CacheIntegrityError',
    'SampleMismatchError',
    'ScenarioDefinition',
    'ReproducibleResultCache',
    'normalize_scenarios',
    'normalize_distributions',
    'normalize_quantiles',
    'build_config',
    'build_manifest',
    'run_reproducible_trade',
    'run_trade',
    'quantile',
    'verify_result',
    'compare_runs',
    'assert_reproducible',
    'register_distribution',
]


class ReproducibilityError(ValueError):
    """Base class for reproducible-trade failures."""


class ConfigurationError(ReproducibilityError):
    """Raised when a trade configuration is invalid or not traceable."""


class EvaluationError(ReproducibilityError):
    """Raised when the metric evaluator returns a non-finite value."""


class ManifestMismatchError(ReproducibilityError):
    """Raised when a result does not match the manifest it is stored under."""


class CacheIntegrityError(ReproducibilityError):
    """Raised when a stored result fails integrity verification."""


class SampleMismatchError(ReproducibilityError):
    """Raised when two runs of the same manifest differ per sample."""

    def __init__(self, mismatches):
        self.mismatches = list(mismatches)
        first = self.mismatches[0] if self.mismatches else None
        super().__init__(
            'runs are not sample-identical: %d difference(s), first: %r'
            % (len(self.mismatches), first)
        )


# ---------------------------------------------------------------------------
# Canonicalization helpers


def _canonical_json(payload):
    try:
        return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError('configuration is not JSON traceable: %s' % exc) from exc


def _digest(payload):
    return hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()


def _derive_seed(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(_canonical_json(part).encode('utf-8'))
        h.update(b'\x1e')
    return int.from_bytes(h.digest(), 'big')


def _json_pure(value, path='$'):
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError('non-finite number at %s' % path)
        return value
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError('non-string key %r at %s' % (key, path))
            out[key] = _json_pure(item, '%s.%s' % (path, key))
        return out
    if isinstance(value, (list, tuple)):
        return [_json_pure(item, '%s[%d]' % (path, i)) for i, item in enumerate(value)]
    raise ConfigurationError(
        'value of type %s at %s is not traceable' % (type(value).__name__, path)
    )


# ---------------------------------------------------------------------------
# Scenario and distribution normalization


@dataclass(frozen=True)
class ScenarioDefinition:
    """One trade-study scenario: a stable id plus JSON-traceable parameters."""

    id: str
    parameters: Mapping = field(default_factory=dict)
    distributions: Mapping = field(default_factory=dict)
    tags: Iterable = ()

    def to_dict(self):
        return {
            'id': self.id,
            'parameters': _json_pure(dict(self.parameters or {}), 'parameters'),
            'distributions': _json_pure(dict(self.distributions or {}), 'distributions'),
            'tags': sorted(str(tag) for tag in (self.tags or ())),
        }


def _normalize_scenario(item, index):
    if isinstance(item, ScenarioDefinition):
        data = item.to_dict()
    elif isinstance(item, Mapping):
        try:
            data = ScenarioDefinition(
                item.get('id', item.get('name')),
                item.get('parameters') or {},
                item.get('distributions') or {},
                item.get('tags') or (),
            ).to_dict()
        except TypeError as exc:
            raise ConfigurationError('scenario %d is not traceable: %s' % (index, exc)) from exc
    else:
        raise ConfigurationError(
            'scenario %d must be a mapping or ScenarioDefinition, got %s'
            % (index, type(item).__name__)
        )
    if not isinstance(data['id'], str) or not data['id']:
        raise ConfigurationError('scenario %d requires a non-empty string id' % index)
    return data


def normalize_scenarios(scenarios):
    """Return the canonical, id-sorted scenario list or raise ConfigurationError."""
    if isinstance(scenarios, (str, bytes, Mapping)) or not isinstance(scenarios, Iterable):
        raise ConfigurationError('scenarios must be an iterable of scenario definitions')
    items = list(scenarios)
    if not items:
        raise ConfigurationError('scenario list must not be empty')
    normalized = [_normalize_scenario(item, i) for i, item in enumerate(items)]
    ids = [row['id'] for row in normalized]
    duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
    if duplicates:
        raise ConfigurationError('duplicate scenario id(s): %s' % ', '.join(duplicates))
    normalized.sort(key=lambda row: row['id'])
    for row in normalized:
        row['distributions'] = _normalize_distribution_map(
            row['distributions'], allow_empty=True, context="scenario '%s' overrides" % row['id']
        )
    return normalized


def _finite_parameter(spec, key, name):
    if key not in spec:
        raise ConfigurationError("distribution '%s' of type '%s' requires '%s'"
                                 % (name, spec['type'], key))
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(
            "distribution '%s' parameter '%s' must be a finite number" % (name, key)
        )
    return float(value)


def _validate_builtin_spec(spec, name):
    kind = spec['type']
    allowed = {'type'}
    if kind == 'constant':
        if 'value' not in spec:
            raise ConfigurationError("distribution '%s' of type 'constant' requires 'value'" % name)
        allowed.add('value')
    elif kind == 'uniform':
        low = _finite_parameter(spec, 'low', name)
        high = _finite_parameter(spec, 'high', name)
        if low > high:
            raise ConfigurationError("distribution '%s' has low > high" % name)
        allowed |= {'low', 'high'}
    elif kind == 'normal':
        _finite_parameter(spec, 'mean', name)
        if _finite_parameter(spec, 'stdev', name) < 0:
            raise ConfigurationError("distribution '%s' has negative stdev" % name)
        allowed |= {'mean', 'stdev'}
    elif kind == 'triangular':
        low = _finite_parameter(spec, 'low', name)
        mode = _finite_parameter(spec, 'mode', name)
        high = _finite_parameter(spec, 'high', name)
        if not low <= mode <= high:
            raise ConfigurationError("distribution '%s' requires low <= mode <= high" % name)
        allowed |= {'low', 'mode', 'high'}
    elif kind == 'choice':
        values = spec.get('values')
        if not isinstance(values, list) or not values:
            raise ConfigurationError("distribution '%s' of type 'choice' requires a non-empty 'values' list" % name)
        allowed.add('values')
    elif kind == 'randint':
        for key in ('low', 'high'):
            value = spec.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(
                    "distribution '%s' of type 'randint' requires integer '%s'" % (name, key)
                )
        if spec['low'] > spec['high']:
            raise ConfigurationError("distribution '%s' has low > high" % name)
        allowed |= {'low', 'high'}
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ConfigurationError(
            "distribution '%s' of type '%s' has unexpected parameter(s): %s"
            % (name, kind, ', '.join(unknown))
        )


def _normalize_distribution_map(distributions, allow_empty=False, context='distributions'):
    if not isinstance(distributions, Mapping):
        raise ConfigurationError('%s must be a mapping of input name to spec' % context)
    if not distributions and not allow_empty:
        raise ConfigurationError('%s must not be empty' % context)
    out = {}
    for name in sorted(distributions):
        if not isinstance(name, str) or not name:
            raise ConfigurationError('%s: names must be non-empty strings' % context)
        spec = distributions[name]
        if not isinstance(spec, Mapping):
            raise ConfigurationError("%s: spec for '%s' must be a mapping" % (context, name))
        spec = _json_pure(dict(spec), "%s['%s']" % (context, name))
        kind = spec.get('type')
        if not isinstance(kind, str) or not kind:
            raise ConfigurationError("%s: spec for '%s' requires a 'type'" % (context, name))
        if kind in _BUILTIN_DISTRIBUTIONS:
            _validate_builtin_spec(spec, name)
        elif kind not in _CUSTOM_DISTRIBUTIONS:
            raise ConfigurationError("unknown distribution type %r for input '%s'" % (kind, name))
        out[name] = spec
    return out


def normalize_distributions(distributions):
    """Return the canonical input-distribution map or raise ConfigurationError."""
    return _normalize_distribution_map(distributions, allow_empty=False)


def normalize_quantiles(quantiles):
    """Return the canonical sorted quantile tuple or raise ConfigurationError."""
    if quantiles is None:
        quantiles = DEFAULT_QUANTILES
    if isinstance(quantiles, (int, float)) and not isinstance(quantiles, bool):
        quantiles = (quantiles,)
    if isinstance(quantiles, (str, bytes)) or not isinstance(quantiles, Iterable):
        raise ConfigurationError('quantiles must be an iterable of probabilities')
    items = list(quantiles)
    if not items:
        raise ConfigurationError('at least one quantile is required')
    out = []
    for q in items:
        if (isinstance(q, bool) or not isinstance(q, (int, float))
                or not math.isfinite(q) or not 0.0 < float(q) < 1.0):
            raise ConfigurationError(
                'quantiles must be finite numbers strictly between 0 and 1, got %r' % (q,)
            )
        out.append(float(q))
    if len(set(out)) != len(out):
        raise ConfigurationError('duplicate quantile levels are not allowed')
    return tuple(sorted(out))


def _normalize_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise ConfigurationError('seed must be an integer or a non-empty string')
    if isinstance(seed, str) and not seed:
        raise ConfigurationError('seed must be an integer or a non-empty string')
    return seed


def _normalize_sample_count(samples):
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ConfigurationError('samples per scenario must be a positive integer')
    return samples


def _normalize_method(method):
    if method not in QUANTILE_METHODS:
        raise ConfigurationError(
            'quantile method must be one of %s, got %r' % (', '.join(QUANTILE_METHODS), method)
        )
    return method


def _normalize_version(distribution_version):
    if not isinstance(distribution_version, str) or not distribution_version:
        raise ConfigurationError('distribution_version must be a non-empty string')
    return distribution_version


# ---------------------------------------------------------------------------
# Distribution sampling


def _draw_constant(rng, spec):
    return spec['value']


def _draw_uniform(rng, spec):
    return rng.uniform(float(spec['low']), float(spec['high']))


def _draw_normal(rng, spec):
    return rng.normalvariate(float(spec['mean']), float(spec['stdev']))


def _draw_triangular(rng, spec):
    return rng.triangular(float(spec['low']), float(spec['high']), float(spec['mode']))


def _draw_choice(rng, spec):
    values = spec['values']
    return values[rng.randrange(len(values))]


def _draw_randint(rng, spec):
    return rng.randint(spec['low'], spec['high'])


_BUILTIN_DISTRIBUTIONS = {
    'constant': _draw_constant,
    'uniform': _draw_uniform,
    'normal': _draw_normal,
    'triangular': _draw_triangular,
    'choice': _draw_choice,
    'randint': _draw_randint,
}

_CUSTOM_DISTRIBUTIONS = {}


def register_distribution(kind, sampler):
    """Register a custom distribution type.

    ``sampler`` is called as ``sampler(rng, spec)`` and must be a pure
    function of the supplied ``rng`` so runs stay reproducible.
    """
    if not isinstance(kind, str) or not kind:
        raise ConfigurationError('distribution type name must be a non-empty string')
    if kind in _BUILTIN_DISTRIBUTIONS:
        raise ConfigurationError('cannot override built-in distribution type %r' % kind)
    if not callable(sampler):
        raise ConfigurationError('distribution sampler must be callable')
    _CUSTOM_DISTRIBUTIONS[kind] = sampler


def _draw(rng, spec, name):
    kind = spec['type']
    if kind in _BUILTIN_DISTRIBUTIONS:
        return _BUILTIN_DISTRIBUTIONS[kind](rng, spec)
    if kind in _CUSTOM_DISTRIBUTIONS:
        value = _CUSTOM_DISTRIBUTIONS[kind](rng, spec)
        return _json_pure(value, "sample for input '%s'" % name)
    raise ConfigurationError("unknown distribution type %r for input '%s'" % (kind, name))


# ---------------------------------------------------------------------------
# Configuration and manifest


def build_config(scenarios, distributions, *, seed, samples=1000, quantiles=None,
                 distribution_version, quantile_method='linear'):
    """Return the canonical, fully traceable trade configuration."""
    return {
        'schema_version': SCHEMA_VERSION,
        'seed': _normalize_seed(seed),
        'samples_per_scenario': _normalize_sample_count(samples),
        'quantiles': list(normalize_quantiles(quantiles)),
        'quantile_method': _normalize_method(quantile_method),
        'distribution_version': _normalize_version(distribution_version),
        'scenarios': normalize_scenarios(scenarios),
        'distributions': normalize_distributions(distributions),
    }


def _manifest_from_config(config):
    return {
        'schema_version': config['schema_version'],
        'seed': config['seed'],
        'samples_per_scenario': config['samples_per_scenario'],
        'quantiles': list(config['quantiles']),
        'quantile_method': config['quantile_method'],
        'distribution_version': config['distribution_version'],
        'scenario_ids': [row['id'] for row in config['scenarios']],
        'scenarios_sha256': _digest(config['scenarios']),
        'distributions_sha256': _digest(config['distributions']),
        'config_sha256': _digest(config),
    }


def build_manifest(scenarios, distributions, *, seed, samples=1000, quantiles=None,
                   distribution_version, quantile_method='linear'):
    """Return the traceability manifest for a trade configuration."""
    config = build_config(
        scenarios, distributions, seed=seed, samples=samples, quantiles=quantiles,
        distribution_version=distribution_version, quantile_method=quantile_method,
    )
    return _manifest_from_config(config)


# ---------------------------------------------------------------------------
# Quantiles and per-scenario summaries


def _quantile_key(q):
    return repr(float(q))


def quantile(values, q, method='linear'):
    """Return the q-quantile of values using a deterministic, documented method."""
    _normalize_method(method)
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0.0 < float(q) < 1.0:
        raise ConfigurationError('q must be strictly between 0 and 1, got %r' % (q,))
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ConfigurationError('quantile requires at least one value')
    n = len(ordered)
    q = float(q)
    if method == 'nearest_rank':
        return ordered[max(0, math.ceil(q * n) - 1)]
    position = q * (n - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _effective_distributions(base, scenario):
    overrides = scenario['distributions']
    if not overrides:
        return base
    merged = dict(base)
    merged.update(overrides)
    return {name: merged[name] for name in sorted(merged)}


def _sample_scenario(config, scenario, distributions):
    seed_payload = {
        'schema_version': config['schema_version'],
        'seed': config['seed'],
        'scenario': scenario,
        'distributions': distributions,
    }
    names = sorted(distributions)
    rows = []
    for index in range(config['samples_per_scenario']):
        rng = random.Random(
            _derive_seed('orbitforge.reproducible_trade.sample', seed_payload, index)
        )
        rows.append({name: _draw(rng, distributions[name], name) for name in names})
    return rows


def _bind_evaluator(evaluate):
    if not callable(evaluate):
        raise ConfigurationError('evaluate must be callable')
    try:
        signature = inspect.signature(evaluate)
    except (TypeError, ValueError):
        return evaluate
    parameters = list(signature.parameters.values())
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters)
    positional = [
        p for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [p for p in positional if p.default is inspect.Parameter.empty]
    if has_varargs or len(positional) >= 2 or len(required) >= 2:
        return evaluate
    return lambda inputs, parameters: evaluate(inputs)


def _evaluate_all(evaluate, scenario, rows):
    values = []
    parameters = scenario['parameters']
    for index, row in enumerate(rows):
        value = evaluate(dict(row), dict(parameters))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise EvaluationError(
                "metric for scenario '%s' sample %d must be a finite number, got %r"
                % (scenario['id'], index, value)
            )
        values.append(float(value))
    return values


def _summarize(scenario_id, rows, values, quantile_levels, method):
    ordered = sorted(values)
    return {
        'scenario_id': scenario_id,
        'sample_count': len(values),
        'samples': rows,
        'values': values,
        'mean': statistics.fmean(values),
        'stdev': statistics.pstdev(values),
        'min': ordered[0],
        'max': ordered[-1],
        'quantiles': {_quantile_key(q): quantile(ordered, q, method) for q in quantile_levels},
        'samples_sha256': _digest({'samples': rows, 'values': values}),
    }


# ---------------------------------------------------------------------------
# Running a trade


def run_reproducible_trade(scenarios, distributions, evaluate, *, seed, samples=1000,
                           quantiles=None, distribution_version, quantile_method='linear',
                           cache=None, reference=None):
    """Run a reproducible uncertainty trade and return the full result artifact.

    ``evaluate`` is called as ``evaluate(inputs, parameters)`` per sample and
    must return a finite number. The returned mapping contains the traceability
    manifest, the canonical configuration and, per scenario, every sampled
    input row and metric value plus the requested quantiles.

    If ``cache`` is given, a verified cached result for the identical
    configuration is returned instead of recomputing. If ``reference`` is
    given, the fresh run is compared sample by sample against it and any
    difference raises :class:`SampleMismatchError`.
    """
    config = build_config(
        scenarios, distributions, seed=seed, samples=samples, quantiles=quantiles,
        distribution_version=distribution_version, quantile_method=quantile_method,
    )
    manifest = _manifest_from_config(config)
    if cache is not None:
        cached = cache.get(manifest)
        if cached is not None:
            if reference is not None:
                assert_reproducible(reference, cached)
            return cached
    evaluator = _bind_evaluator(evaluate)
    results = []
    for scenario in config['scenarios']:
        effective = _effective_distributions(config['distributions'], scenario)
        rows = _sample_scenario(config, scenario, effective)
        values = _evaluate_all(evaluator, scenario, rows)
        results.append(
            _summarize(scenario['id'], rows, values, config['quantiles'], config['quantile_method'])
        )
    result = {
        'manifest': manifest,
        'config': config,
        'scenarios': results,
    }
    result['result_sha256'] = _digest(result)
    if reference is not None:
        assert_reproducible(reference, result)
    if cache is not None:
        cache.put(manifest, result)
    return result


run_trade = run_reproducible_trade


# ---------------------------------------------------------------------------
# Verification and comparison


def verify_result(result):
    """Verify a result artifact's internal consistency; raise CacheIntegrityError."""
    if not isinstance(result, Mapping):
        raise CacheIntegrityError('result must be a mapping')
    for key in ('manifest', 'config', 'scenarios', 'result_sha256'):
        if key not in result:
            raise CacheIntegrityError("result is missing '%s'" % key)
    config = result['config']
    if not isinstance(config, Mapping) or config.get('schema_version') != SCHEMA_VERSION:
        raise CacheIntegrityError('embedded configuration has an unsupported schema version')
    try:
        expected_manifest = _manifest_from_config(config)
    except (KeyError, TypeError) as exc:
        raise CacheIntegrityError('embedded configuration is incomplete: %r' % (exc,)) from exc
    if result['manifest'] != expected_manifest:
        raise CacheIntegrityError('manifest does not match the embedded configuration')
    body = {key: result[key] for key in ('manifest', 'config', 'scenarios')}
    if _digest(body) != result['result_sha256']:
        raise CacheIntegrityError('result digest mismatch')
    entries = result['scenarios']
    if not isinstance(entries, list):
        raise CacheIntegrityError('scenario results must be a list')
    expected_ids = [row['id'] for row in config['scenarios']]
    actual_ids = [entry.get('scenario_id') for entry in entries]
    if actual_ids != expected_ids:
        raise CacheIntegrityError('scenario results do not match the manifest scenario list')
    for entry in entries:
        sid = entry.get('scenario_id')
        rows = entry.get('samples')
        values = entry.get('values')
        if not isinstance(rows, list) or not isinstance(values, list) or len(rows) != len(values):
            raise CacheIntegrityError("sample rows and values diverge for scenario '%s'" % sid)
        if len(values) != config['samples_per_scenario']:
            raise CacheIntegrityError("sample count mismatch for scenario '%s'" % sid)
        if _digest({'samples': rows, 'values': values}) != entry.get('samples_sha256'):
            raise CacheIntegrityError("sample digest mismatch for scenario '%s'" % sid)
        ordered = sorted(values)
        expected_stats = {
            'sample_count': len(values),
            'mean': statistics.fmean(values),
            'stdev': statistics.pstdev(values),
            'min': ordered[0],
            'max': ordered[-1],
        }
        for key, expected in expected_stats.items():
            if entry.get(key) != expected:
                raise CacheIntegrityError(
                    "statistic '%s' mismatch for scenario '%s'" % (key, sid)
                )
        expected_quantiles = {
            _quantile_key(q): quantile(ordered, q, config['quantile_method'])
            for q in config['quantiles']
        }
        if entry.get('quantiles') != expected_quantiles:
            raise CacheIntegrityError("quantiles mismatch for scenario '%s'" % sid)
    return True


def compare_runs(first, second, limit=50):
    """Return a list of per-sample differences between two run artifacts."""
    mismatches = []

    def add(entry):
        if len(mismatches) < limit:
            mismatches.append(entry)

    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        add({'scope': 'result', 'detail': 'both runs must be result mappings'})
        return mismatches
    if first.get('manifest') != second.get('manifest'):
        add({'scope': 'manifest', 'detail': 'manifests differ'})
    first_by_id = {e.get('scenario_id'): e for e in first.get('scenarios') or []}
    second_by_id = {e.get('scenario_id'): e for e in second.get('scenarios') or []}
    for sid in sorted(set(first_by_id) | set(second_by_id)):
        a = first_by_id.get(sid)
        b = second_by_id.get(sid)
        if a is None:
            add({'scope': 'scenario', 'scenario_id': sid, 'detail': 'missing in first run'})
            continue
        if b is None:
            add({'scope': 'scenario', 'scenario_id': sid, 'detail': 'missing in second run'})
            continue
        a_rows = a.get('samples') or []
        b_rows = b.get('samples') or []
        if len(a_rows) != len(b_rows):
            add({'scope': 'scenario', 'scenario_id': sid,
             'detail': 'sample counts differ: %d vs %d' % (len(a_rows), len(b_rows))})
        for index, (ra, rb) in enumerate(zip(a_rows, b_rows)):
            if ra != rb:
                add({'scope': 'sample', 'scenario_id': sid, 'sample_index': index,
                     'first': ra, 'second': rb})
        a_values = a.get('values') or []
        b_values = b.get('values') or []
        for index, (va, vb) in enumerate(zip(a_values, b_values)):
            if va != vb:
                add({'scope': 'value', 'scenario_id': sid, 'sample_index': index,
                     'first': va, 'second': vb})
    return mismatches


def assert_reproducible(first, second):
    """Raise SampleMismatchError unless two runs are sample-identical."""
    mismatches = compare_runs(first, second)
    if mismatches:
        raise SampleMismatchError(mismatches)
    return True


# ---------------------------------------------------------------------------
# Result cache


def _manifest_key(manifest):
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get('config_sha256'), str):
        raise ConfigurationError('manifest must be a mapping with a config_sha256 digest')
    return manifest['config_sha256']


class ReproducibleResultCache:
    """Content-addressed result cache keyed by configuration digest.

    A hit is only possible for the identical configuration (scenario list,
    seed, distribution version, quantiles, ...). Stored results are verified
    on every access; tampered or conflicting entries raise explicit errors
    instead of silently serving stale data.
    """

    def __init__(self):
        self._records = {}

    def __len__(self):
        return len(self._records)

    def get(self, manifest):
        key = _manifest_key(manifest)
        record = self._records.get(key)
        if record is None:
            return None
        if record['manifest'] != manifest:
            raise ManifestMismatchError(
                'cached result manifest does not match the requested configuration %s'
                % key[:12]
            )
        verify_result(record)
        return copy.deepcopy(record)

    def put(self, manifest, result):
        key = _manifest_key(manifest)
        if not isinstance(result, Mapping) or result.get('manifest') != manifest:
            raise ManifestMismatchError(
                'result was produced under a different manifest than configuration %s'
                % key[:12]
            )
        verify_result(result)
        existing = self._records.get(key)
        if existing is not None and existing != result:
            raise CacheIntegrityError(
                'conflicting results for configuration %s' % key[:12]
            )
        self._records[key] = copy.deepcopy(result)
        return key

    def invalidate(self, manifest=None):
        if manifest is None:
            keys = sorted(self._records)
            self._records.clear()
            return keys
        key = _manifest_key(manifest)
        return [key] if self._records.pop(key, None) is not None else []
