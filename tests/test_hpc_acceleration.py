"""Tests unitaires du Module 6 : profiling, Numba JIT et parallelisme Joblib."""

from __future__ import annotations

import numpy as np
from src.hpc_acceleration import (
    ProfilingResult,
    fast_local_filter,
    naive_local_filter,
    parameter_sweep,
    profile_filter,
)


def test_naive_and_jit_filters_agree_numerically() -> None:
    rng = np.random.default_rng(0)
    grid = rng.standard_normal((12, 12))

    naive_result = naive_local_filter(grid, radius=1)
    jit_result = fast_local_filter(grid, radius=1)

    assert np.allclose(naive_result, jit_result, atol=1e-10)


def test_profile_filter_returns_positive_timings_and_speedup() -> None:
    rng = np.random.default_rng(1)
    grid = rng.standard_normal((20, 20))

    result = profile_filter(grid, radius=1, number=1)

    assert result.naive_seconds >= 0.0
    assert result.jit_seconds >= 0.0
    assert result.speedup > 0.0


def test_profiling_result_speedup_handles_zero_jit_time() -> None:
    result = ProfilingResult(naive_seconds=1.0, jit_seconds=0.0)
    assert result.speedup == float("inf")


def test_parameter_sweep_evaluates_all_combinations() -> None:
    def evaluator(c: float, nu: float) -> float:
        return c + nu

    results = parameter_sweep([1.0, 2.0], [0.1, 0.2], evaluator, n_jobs=2)

    assert len(results) == 4
    values = {(c, nu): value for c, nu, value in results}
    assert values[(1.0, 0.1)] == 1.1
    assert values[(2.0, 0.2)] == 2.2
