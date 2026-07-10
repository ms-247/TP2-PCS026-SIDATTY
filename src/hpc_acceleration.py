"""Profiling, acceleration JIT (Numba) et parallelisme multi-processus (Joblib).

Module 6 : un operateur de filtrage local (moyenne glissante 2D) est d'abord
implemente naivement avec des boucles imbriquees Python, puis accelere via
Numba (@njit, parallel=True, fastmath=True). Un balayage de parametres (c, nu)
est ensuite distribue sur plusieurs processus avec Joblib.
"""

from __future__ import annotations

import timeit
from typing import Callable

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from numba import njit, prange

FloatArray = npt.NDArray[np.float64]


def naive_local_filter(grid: FloatArray, radius: int = 1) -> FloatArray:
    """Filtre de moyenne glissante 2D, implementation naive en boucles imbriquees."""
    n_rows, n_cols = grid.shape
    output = np.zeros_like(grid)
    for i in range(n_rows):
        for j in range(n_cols):
            i_min, i_max = max(0, i - radius), min(n_rows, i + radius + 1)
            j_min, j_max = max(0, j - radius), min(n_cols, j + radius + 1)
            window_sum = 0.0
            count = 0
            for wi in range(i_min, i_max):
                for wj in range(j_min, j_max):
                    window_sum += grid[wi, wj]
                    count += 1
            output[i, j] = window_sum / count
    return output


@njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
def fast_local_filter(grid: FloatArray, radius: int) -> FloatArray:
    """Version JIT-compilee (Numba) du filtre de moyenne glissante 2D.

    parallel=True active la vectorisation et la repartition des iterations
    de la boucle externe (prange) sur plusieurs coeurs CPU. fastmath=True
    autorise le compilateur a reordonner les operations flottantes en
    relachant la stricte conformite IEEE 754 (voir README pour les risques).
    """
    n_rows, n_cols = grid.shape
    output = np.zeros_like(grid)
    for i in prange(n_rows):
        for j in range(n_cols):
            i_min = max(0, i - radius)
            i_max = min(n_rows, i + radius + 1)
            j_min = max(0, j - radius)
            j_max = min(n_cols, j + radius + 1)
            window_sum = 0.0
            count = 0
            for wi in range(i_min, i_max):
                for wj in range(j_min, j_max):
                    window_sum += grid[wi, wj]
                    count += 1
            output[i, j] = window_sum / count
    return output


class ProfilingResult:
    """Resultat de comparaison de performance naive vs Numba JIT."""

    def __init__(self, naive_seconds: float, jit_seconds: float) -> None:
        self.naive_seconds = naive_seconds
        self.jit_seconds = jit_seconds

    @property
    def speedup(self) -> float:
        if self.jit_seconds <= 0.0:
            return float("inf")
        return self.naive_seconds / self.jit_seconds


def profile_filter(grid: FloatArray, radius: int = 1, number: int = 3) -> ProfilingResult:
    """Mesure via timeit le temps d'execution naif vs accelere (goulot d'etranglement)."""
    fast_local_filter(grid[:4, :4].copy(), radius)  # warm-up : compilation JIT hors chronometrage

    naive_seconds = timeit.timeit(lambda: naive_local_filter(grid, radius), number=number) / number
    jit_seconds = timeit.timeit(lambda: fast_local_filter(grid, radius), number=number) / number
    return ProfilingResult(naive_seconds=naive_seconds, jit_seconds=jit_seconds)


def _evaluate_parameter_pair(
    c_value: float, nu_value: float, evaluator: Callable[[float, float], float]
) -> tuple[float, float, float]:
    """Evalue une combinaison (c, nu) unique, unite de travail parallelisable."""
    result = evaluator(c_value, nu_value)
    return c_value, nu_value, result


def parameter_sweep(
    c_values: list[float],
    nu_values: list[float],
    evaluator: Callable[[float, float], float],
    n_jobs: int = 2,
) -> list[tuple[float, float, float]]:
    """Explore l'espace de parametres (c, nu) en parallele multi-processus via Joblib."""
    pairs = [(c, nu) for c in c_values for nu in nu_values]
    results: list[tuple[float, float, float]] = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_parameter_pair)(c, nu, evaluator) for c, nu in pairs
    )
    return results
