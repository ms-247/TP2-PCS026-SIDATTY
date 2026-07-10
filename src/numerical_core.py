"""Manipulation bas niveau de ndarray, ingestion vectorisee et analyse de stabilite.

Module 4 : grille spatio-temporelle, layout memoire (C/F-contiguous), vues vs
copies, ingestion Polars Lazy et broadcasting.
Module 5 : conditionnement numerique (matrice de Hilbert), sensibilite aux
types flottants (float16/32/64) et validation robuste par np.isclose.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import polars as pl

FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Module 4.1 : grille spatio-temporelle et layout memoire
# ---------------------------------------------------------------------------


def create_spatiotemporal_grid(
    n: int,
    m: int,
    x_range: tuple[float, float] = (-5.0, 5.0),
    t_range: tuple[float, float] = (0.0, 2.0),
) -> tuple[FloatArray, FloatArray]:
    """Cree une grille bidimensionnelle (X, T) de taille N x M (C-contiguous)."""
    x = np.linspace(x_range[0], x_range[1], n, dtype=np.float64)
    t = np.linspace(t_range[0], t_range[1], m, dtype=np.float64)
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    return np.ascontiguousarray(x_grid), np.ascontiguousarray(t_grid)


class ArrayDescriptor(TypedDict):
    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...]
    c_contiguous: bool
    f_contiguous: bool


def describe_array(arr: FloatArray) -> ArrayDescriptor:
    """Explore et retourne explicitement shape, dtype et strides d'un tableau."""
    return ArrayDescriptor(
        shape=arr.shape,
        dtype=str(arr.dtype),
        strides=arr.strides,
        c_contiguous=bool(arr.flags["C_CONTIGUOUS"]),
        f_contiguous=bool(arr.flags["F_CONTIGUOUS"]),
    )


def to_fortran_order(arr: FloatArray) -> FloatArray:
    """Retourne une copie de arr en agencement memoire F-contiguous (colonne)."""
    return np.asfortranarray(arr)


def row_reduction_benchmark(arr: FloatArray) -> float:
    """Somme par ligne (parcours du dernier axe) : favorable au layout C."""
    return float(np.sum(arr, axis=1).sum())


def column_reduction_benchmark(arr: FloatArray) -> float:
    """Somme par colonne (parcours du premier axe) : favorable au layout F."""
    return float(np.sum(arr, axis=0).sum())


def slice_is_view(arr: FloatArray, row_slice: slice, col_slice: slice) -> tuple[FloatArray, bool]:
    """Realise un slicing et determine via .base s'il s'agit d'une vue ou d'une copie."""
    sliced = arr[row_slice, col_slice]
    is_view = sliced.base is not None
    return sliced, is_view


def fancy_index_is_copy(
    arr: FloatArray, row_indices: npt.NDArray[np.intp]
) -> tuple[FloatArray, bool]:
    """Realise une indexation avancee (fancy indexing), toujours une copie effective."""
    selected = arr[row_indices]
    is_view = selected.base is not None
    return selected, is_view


# ---------------------------------------------------------------------------
# Module 4.2 : ingestion Polars Lazy + broadcasting
# ---------------------------------------------------------------------------


def ingest_sensor_coordinates(
    path: Path,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
) -> tuple[FloatArray, FloatArray]:
    """Scanne un fichier Parquet de capteurs en mode Lazy et filtre les coordonnees valides.

    N'encombre pas la RAM : le filtrage est planifie via l'API Lazy de Polars
    et n'est materialise (collect) qu'une fois le plan optimise.
    """
    lazy_frame = pl.scan_parquet(path)
    filtered = lazy_frame.filter(
        pl.col("latitude").is_between(lat_range[0], lat_range[1])
        & pl.col("longitude").is_between(lon_range[0], lon_range[1])
    ).select(["latitude", "longitude"])

    collected = filtered.collect()
    latitude = collected["latitude"].to_numpy().astype(np.float64)
    longitude = collected["longitude"].to_numpy().astype(np.float64)
    return latitude, longitude


def apply_residual_vectorized(
    f_func: Callable[[FloatArray, FloatArray], FloatArray],
    x_coords: FloatArray,
    t_coords: FloatArray,
) -> FloatArray:
    """Applique la fonction residuelle f(x,t) sur des coordonnees via broadcasting NumPy.

    Aucune boucle for : x_coords et t_coords sont diffuses (broadcast) l'un
    contre l'autre pour produire une evaluation instantanee et vectorisee.
    """
    x_col = x_coords.reshape(-1, 1)
    t_row = t_coords.reshape(1, -1)
    result = f_func(x_col, t_row)
    return np.asarray(result, dtype=np.float64)


# ---------------------------------------------------------------------------
# Module 5 : stabilite, conditionnement et analyse des erreurs IEEE 754
# ---------------------------------------------------------------------------


def build_hilbert_matrix(n: int) -> FloatArray:
    """Construit la matrice de Hilbert H_ij = 1 / (i + j - 1), tristement mal conditionnee."""
    i = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
    j = np.arange(1, n + 1, dtype=np.float64).reshape(1, -1)
    return 1.0 / (i + j - 1.0)


def condition_number(matrix: FloatArray) -> float:
    """Evalue le nombre de conditionnement kappa(A) = ||A|| . ||A^-1||."""
    return float(np.linalg.cond(matrix))


class PrecisionErrors(TypedDict):
    float16: float
    float32: float
    float64: float


def reconstruction_error_by_precision(matrix: FloatArray, rhs: FloatArray) -> PrecisionErrors:
    """Compare l'erreur de reconstruction ||Ax - b|| en float16, float32 et float64."""
    errors: dict[str, float] = {}
    for dtype in (np.float16, np.float32, np.float64):
        a_cast = matrix.astype(dtype)
        b_cast = rhs.astype(dtype)
        a64, b64 = a_cast.astype(np.float64), b_cast.astype(np.float64)
        solution = np.linalg.solve(a64, b64).astype(dtype)
        residual = a64 @ solution.astype(np.float64) - b64
        errors[np.dtype(dtype).name] = float(np.linalg.norm(residual))
    return PrecisionErrors(
        float16=errors["float16"], float32=errors["float32"], float64=errors["float64"]
    )


def perturb_and_propagate(
    matrix: FloatArray, rhs: FloatArray, epsilon: float = 1e-7, seed: int = 42
) -> tuple[FloatArray, FloatArray, float]:
    """Perturbe b d'ordre epsilon et mesure l'amplification d'erreur sur la solution alpha.

    Retourne (alpha, alpha_perturbe, erreur_relative_solution). L'amplification
    est bornee par kappa(A) * erreur_relative_b (theorie de perturbation lineaire).
    """
    rng = np.random.default_rng(seed)
    perturbation = epsilon * rng.standard_normal(rhs.shape)
    rhs_perturbed = rhs + perturbation

    alpha: FloatArray = np.linalg.solve(matrix, rhs).astype(np.float64)
    alpha_perturbed: FloatArray = np.linalg.solve(matrix, rhs_perturbed).astype(np.float64)

    relative_error = float(np.linalg.norm(alpha_perturbed - alpha) / np.linalg.norm(alpha))
    return alpha, alpha_perturbed, relative_error


def compute_residual(matrix: FloatArray, alpha: FloatArray, rhs: FloatArray) -> float:
    """Calcule le residu strict r = ||A.alpha - b||."""
    return float(np.linalg.norm(matrix @ alpha - rhs))


def validate_solution(
    matrix: FloatArray, alpha: FloatArray, rhs: FloatArray, atol: float = 1e-6, rtol: float = 1e-5
) -> bool:
    """Valide Aα ≈ b de maniere robuste avec np.isclose (jamais d'egalite stricte ==).

    Les egalites strictes sont proscrites car l'arithmetique IEEE 754 en
    virgule flottante accumule des erreurs d'arrondi lors des produits
    matriciels ; atol/rtol absorbent ce bruit numerique attendu.
    """
    reconstructed = matrix @ alpha
    return bool(np.all(np.isclose(reconstructed, rhs, atol=atol, rtol=rtol)))
