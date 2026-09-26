from __future__ import annotations
import hashlib
import json
import random
import statistics

def sample_metric(fn, samplers, n=1000, seed=1):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        vals.append(fn(**{k: s(rng) for k, s in samplers.items()}))
    vals.sort()
    return {'mean': statistics.fmean(vals), 'stdev': statistics.pstdev(vals), 'p05': vals[int(0.05 * (n - 1))], 'p50': vals[int(0.5 * (n - 1))], 'p95': vals[int(0.95 * (n - 1))]}

def sample_seed(seed, index):
    # Per-sample substream seed: depends only on (seed, index), never on
    # execution order, sample count or dict iteration order.
    raw = hashlib.sha256(f'{seed}:{index}'.encode()).digest()
    return int.from_bytes(raw[:8], 'big')

def draw_samples(fn, samplers, n, seed):
    if n < 1:
        raise ValueError(f'n must be >= 1, got {n}')
    out = []
    for i in range(n):
        rng = random.Random(sample_seed(seed, i))
        out.append(fn(**{k: s(rng) for k, s in sorted(samplers.items())}))
    return out

def quantile(sorted_vals, q):
    # Linear interpolation on (n - 1) * q; exact and platform independent.
    if not sorted_vals:
        raise ValueError('quantile of an empty sample')
    if not 0.0 <= q <= 1.0:
        raise ValueError(f'quantile must be in [0, 1], got {q}')
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

def samples_digest(samples):
    raw = json.dumps(samples, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

def summarize(samples, quantiles=(0.05, 0.5, 0.95)):
    vals = sorted(samples)
    return {
        'n': len(vals),
        'mean': statistics.fmean(vals),
        'stdev': statistics.pstdev(vals),
        'quantiles': {repr(float(q)): quantile(vals, q) for q in quantiles},
        'samples_sha256': samples_digest(samples),
    }
