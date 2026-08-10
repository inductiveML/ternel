from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def nearest_rank_from_histogram(histogram: npt.ArrayLike, percentile: int) -> int:
    values = np.asarray(histogram, dtype=np.int64)
    total = int(values.sum())
    if total == 0:
        return 0
    rank = max(1, math.ceil(percentile / 100 * total))
    return int(np.searchsorted(np.cumsum(values), rank, side="left"))


def summarize_symbols(symbol_counts: npt.ArrayLike, nonzero_histogram: npt.ArrayLike) -> dict:
    counts = np.asarray(symbol_counts, dtype=np.int64)
    histogram = np.asarray(nonzero_histogram, dtype=np.int64)
    total_weights = int(counts.sum())
    total_groups = int(histogram.sum())
    if total_weights == 0:
        probabilities = np.zeros_like(counts, dtype=np.float64)
    else:
        probabilities = counts.astype(np.float64) / total_weights
    entropy = float(-sum(p * math.log2(p) for p in probabilities[:3] if p > 0))
    result = {
        "symbol_counts": {
            "-1": int(counts[0]) if counts.size > 0 else 0,
            "0": int(counts[1]) if counts.size > 1 else 0,
            "+1": int(counts[2]) if counts.size > 2 else 0,
            "+2": int(counts[3]) if counts.size > 3 else 0,
        },
        "fractions": {
            "-1": float(probabilities[0]) if probabilities.size > 0 else 0.0,
            "0": float(probabilities[1]) if probabilities.size > 1 else 0.0,
            "+1": float(probabilities[2]) if probabilities.size > 2 else 0.0,
            "+2": float(probabilities[3]) if probabilities.size > 3 else 0.0,
        },
        "zero_density": float(probabilities[1]) if probabilities.size > 1 else 0.0,
        "nonzero_density": 1.0 - (float(probabilities[1]) if probabilities.size > 1 else 0.0),
        "entropy_bits_per_symbol": entropy,
        "nonzeros_per_group": {
            f"p{percentile}": nearest_rank_from_histogram(histogram, percentile)
            for percentile in PERCENTILES
        },
        "group_threshold_fractions": {},
        "groups": total_groups,
        "weights": total_weights,
    }
    cumulative = np.cumsum(histogram)
    for threshold in (32, 48, 64, 80):
        count = int(cumulative[threshold]) if cumulative.size > threshold else total_groups
        result["group_threshold_fractions"][f"le_{threshold}"] = (
            count / total_groups if total_groups else 0.0
        )
    return result

