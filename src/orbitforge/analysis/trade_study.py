from __future__ import annotations
import hashlib
import json

from orbitforge.core.errors import ReproducibilityError, ValidationError
from orbitforge.analysis import monte_carlo

SCHEMA_VERSION = 1

# Distribution registry: (name, version) -> factory(params) -> sampler(rng).
# The version is part of the manifest fingerprint, so a changed distribution
# implementation must be registered under a new version instead of being
# edited in place; old manifests then keep resolving to their original draws.
_DISTRIBUTIONS = {}

def register_distribution(name, version, factory):
    key = (name, version)
    if key in _DISTRIBUTIONS:
        raise ValidationError(f'distribution {name} v{version} already registered')
    _DISTRIBUTIONS[key] = factory

def sampler_for(spec):
    name = spec.get('name')
    version = spec.get('version', 1)
    factory = _DISTRIBUTIONS.get((name, version))
    if factory is None:
        raise ValidationError(f'unknown distribution {name!r} version {version!r}')
    return factory(spec.get('params', {}))

def _uniform(params):
    return lambda rng: rng.uniform(params['low'], params['high'])

def _normal(params):
    return lambda rng: rng.gauss(params['mu'], params['sigma'])

def _lognormal(params):
    return lambda rng: rng.lognormvariate(params['mu'], params['sigma'])

def _triangular(params):
    return lambda rng: rng.triangular(params['low'], params['high'], params['mode'])

def _choice(params):
    values = list(params['values'])
    return lambda rng: values[rng.randrange(len(values))]

def _constant(params):
    return lambda rng: params['value']

register_distribution('uniform', 1, _uniform)
register_distribution('normal', 1, _normal)
register_distribution('lognormal', 1, _lognormal)
register_distribution('triangular', 1, _triangular)
register_distribution('choice', 1, _choice)
register_distribution('constant', 1, _constant)

def _validate_scenario(scenario):
    required = ('name', 'metric', 'n', 'seed', 'distributions')
    missing = [k for k in required if k not in scenario]
    if missing:
        raise ValidationError(f'scenario missing keys: {missing}')
    if not isinstance(scenario['n'], int) or scenario['n'] < 1:
        raise ValidationError(f"scenario {scenario['name']!r}: n must be a positive int")
    if not isinstance(scenario['seed'], int):
        raise ValidationError(f"scenario {scenario['name']!r}: seed must be an int")
    for arg, spec in scenario['distributions'].items():
        if 'name' not in spec:
            raise ValidationError(f"scenario {scenario['name']!r}: distribution for {arg!r} has no name")
        if (spec['name'], spec.get('version', 1)) not in _DISTRIBUTIONS:
            raise ValidationError(f"scenario {scenario['name']!r}: unknown distribution {spec['name']!r} version {spec.get('version', 1)!r} for {arg!r}")

def build_manifest(study, scenarios, quantiles=(0.05, 0.5, 0.95)):
    names = [s.get('name') for s in scenarios]
    if len(set(names)) != len(names):
        raise ValidationError(f'duplicate scenario names: {sorted(names)}')
    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            raise ValidationError(f'quantile must be in [0, 1], got {q}')
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'study': study,
        'quantiles': sorted(float(q) for q in quantiles),
        'scenarios': [dict(s) for s in scenarios],
    }
    for scenario in manifest['scenarios']:
        _validate_scenario(scenario)
    try:
        json.dumps(manifest, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f'manifest is not canonically serializable: {exc}')
    return manifest

def _canonical(manifest):
    # Scenario order carries no meaning; canonicalize so reordering the same
    # manifest yields the same fingerprint and the same results.
    payload = dict(manifest)
    payload['scenarios'] = sorted(manifest['scenarios'], key=lambda s: s['name'])
    return payload

def manifest_fingerprint(manifest, software_version='', metric_versions=None):
    payload = {
        'manifest': _canonical(manifest),
        'software_version': software_version,
        'metric_versions': metric_versions or {},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

def _run_scenario(manifest, scenario, metrics):
    metric = scenario['metric']
    if metric not in metrics:
        raise ValidationError(f"scenario {scenario['name']!r}: unknown metric {metric!r}")
    samplers = {arg: sampler_for(spec) for arg, spec in scenario['distributions'].items()}
    samples = monte_carlo.draw_samples(metrics[metric], samplers, scenario['n'], scenario['seed'])
    summary = monte_carlo.summarize(samples, manifest['quantiles'])
    return {'name': scenario['name'], 'metric': metric, 'seed': scenario['seed'], **summary}

def run_manifest(manifest, metrics, software_version='', metric_versions=None):
    scenarios = sorted(manifest['scenarios'], key=lambda s: s['name'])
    return {
        'fingerprint': manifest_fingerprint(manifest, software_version, metric_versions),
        'study': manifest['study'],
        'schema_version': manifest.get('schema_version', SCHEMA_VERSION),
        'scenarios': [_run_scenario(manifest, s, metrics) for s in scenarios],
    }

def verify_manifest(manifest, recorded, metrics, software_version='', metric_versions=None):
    fingerprint = manifest_fingerprint(manifest, software_version, metric_versions)
    if recorded.get('fingerprint') != fingerprint:
        raise ReproducibilityError(
            f"fingerprint mismatch: manifest hashes to {fingerprint}, recorded result claims {recorded.get('fingerprint')}")
    fresh = run_manifest(manifest, metrics, software_version, metric_versions)
    recorded_scenarios = {s['name']: s for s in recorded.get('scenarios', [])}
    for scenario in fresh['scenarios']:
        name = scenario['name']
        if name not in recorded_scenarios:
            raise ReproducibilityError(f"scenario {name!r} missing from recorded result")
        old = recorded_scenarios[name]
        if old.get('seed') != scenario['seed']:
            raise ReproducibilityError(f"scenario {name!r}: seed changed {old.get('seed')} -> {scenario['seed']}")
        if old.get('samples_sha256') != scenario['samples_sha256']:
            raise ReproducibilityError(
                f"scenario {name!r}: per-sample stream mismatch "
                f"(recorded {old.get('samples_sha256')}, recomputed {scenario['samples_sha256']})")
        if old.get('quantiles') != scenario['quantiles']:
            raise ReproducibilityError(
                f"scenario {name!r}: quantiles mismatch (recorded {old.get('quantiles')}, recomputed {scenario['quantiles']})")
    extra = set(recorded_scenarios) - {s['name'] for s in fresh['scenarios']}
    if extra:
        raise ReproducibilityError(f'recorded result has unknown scenarios: {sorted(extra)}')
    return True

class ResultStore:
    # Cache keyed solely by the manifest fingerprint: a different
    # configuration can never hit a result recorded under another config.
    def __init__(self):
        self._by_fingerprint = {}

    def get(self, fingerprint):
        return self._by_fingerprint.get(fingerprint)

    def put(self, result):
        fingerprint = result.get('fingerprint')
        if not fingerprint:
            raise ValidationError('result has no fingerprint')
        self._by_fingerprint[fingerprint] = result
        return fingerprint

    def __len__(self):
        return len(self._by_fingerprint)

def run_cached(manifest, metrics, store, software_version='', metric_versions=None):
    fingerprint = manifest_fingerprint(manifest, software_version, metric_versions)
    cached = store.get(fingerprint)
    if cached is not None:
        verify_manifest(manifest, cached, metrics, software_version, metric_versions)
        return cached
    result = run_manifest(manifest, metrics, software_version, metric_versions)
    store.put(result)
    return result
