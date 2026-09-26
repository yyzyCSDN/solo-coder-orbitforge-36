from __future__ import annotations
import random, statistics

from orbitforge.analysis.reproducible_trade import (  # noqa: F401
    CacheIntegrityError,
    ConfigurationError,
    EvaluationError,
    ManifestMismatchError,
    ReproducibilityError,
    ReproducibleResultCache,
    SampleMismatchError,
    ScenarioDefinition,
    assert_reproducible,
    build_config,
    build_manifest,
    compare_runs,
    normalize_distributions,
    normalize_quantiles,
    normalize_scenarios,
    quantile,
    register_distribution,
    run_reproducible_trade,
    run_trade,
    verify_result,
)


def sample_metric(fn, samplers, n=1000, seed=1):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(fn(**{k: s(rng) for k, s in samplers.items()}))
    vals.sort()
    return {'mean': statistics.fmean(vals), 'stdev': statistics.pstdev(vals), 'p05': vals[int(0.05 * (n - 1))], 'p50': vals[int(0.5 * (n - 1))], 'p95': vals[int(0.95 * (n - 1))]}
