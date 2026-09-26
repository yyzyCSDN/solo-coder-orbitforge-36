# OrbitForge Mission Lab

OrbitForge is a Python library and small HTTP service for spacecraft mission
analysis: orbital mechanics, mission geometry and resource products.

## Quick start

```bash
python -m pip install -e '.[test]'
PYTHONPATH=src python -m pytest -q
orbitforge
```

The service listens on `127.0.0.1:8080` by default. `GET /live` and `GET /ready`
report service state, and the analysis endpoints live under `/v1/`.

## Reproducible uncertainty trades

`orbitforge.analysis.reproducible_trade` (also re-exported from
`orbitforge.analysis.monte_carlo`) upgrades one-shot Monte Carlo statistics
into a traceable trade artifact:

```python
from orbitforge.analysis.reproducible_trade import run_reproducible_trade, ReproducibleResultCache

result = run_reproducible_trade(
    [{'id': 'leo', 'parameters': {'alt_km': 500}}],
    {'mass_kg': {'type': 'uniform', 'low': 900.0, 'high': 1100.0}},
    lambda inputs, params: inputs['mass_kg'] * 0.001 + params['alt_km'],
    seed=20260926,
    samples=1000,
    quantiles=(0.05, 0.5, 0.95),
    distribution_version='input-distributions-2026-09',
    cache=ReproducibleResultCache(),
)
```

The manifest records the scenario list, seed, distribution version and
quantiles behind a single `config_sha256`. Every scenario keeps its per-sample
inputs and metric values, so rerunning the same manifest reproduces them
exactly (use `reference=` or `assert_reproducible` to enforce it), cached
results are verified on every hit, and any inconsistency raises an explicit
`ReproducibilityError` subclass.
