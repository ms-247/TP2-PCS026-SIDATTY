"""Tests unitaires des Modules 4 et 5 : NumPy bas niveau, ingestion et stabilite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.numerical_core import (
    apply_residual_vectorized,
    build_hilbert_matrix,
    column_reduction_benchmark,
    compute_residual,
    condition_number,
    create_spatiotemporal_grid,
    describe_array,
    fancy_index_is_copy,
    ingest_sensor_coordinates,
    perturb_and_propagate,
    reconstruction_error_by_precision,
    row_reduction_benchmark,
    slice_is_view,
    to_fortran_order,
    validate_solution,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_sensors" / "sensors.parquet"


# --- Module 4.1 : grille et layout memoire -----------------------------------


def test_create_spatiotemporal_grid_has_expected_shape() -> None:
    x_grid, t_grid = create_spatiotemporal_grid(10, 20)
    assert x_grid.shape == (10, 20)
    assert t_grid.shape == (10, 20)
    assert x_grid.flags["C_CONTIGUOUS"]


def test_describe_array_reports_shape_dtype_strides() -> None:
    x_grid, _ = create_spatiotemporal_grid(4, 5)
    descriptor = describe_array(x_grid)
    assert descriptor["shape"] == (4, 5)
    assert descriptor["dtype"] == "float64"
    assert descriptor["c_contiguous"] is True
    assert descriptor["f_contiguous"] is False


def test_fortran_order_changes_memory_layout_not_values() -> None:
    x_grid, _ = create_spatiotemporal_grid(6, 7)
    f_grid = to_fortran_order(x_grid)
    assert f_grid.flags["F_CONTIGUOUS"]
    assert np.array_equal(f_grid, x_grid)


def test_row_and_column_reduction_are_numerically_equivalent() -> None:
    x_grid, _ = create_spatiotemporal_grid(8, 8)
    assert row_reduction_benchmark(x_grid) == pytest.approx(column_reduction_benchmark(x_grid))


def test_slice_is_view_returns_true_for_basic_slicing() -> None:
    x_grid, _ = create_spatiotemporal_grid(10, 10)
    sliced, is_view = slice_is_view(x_grid, slice(0, 5), slice(0, 5))
    assert is_view is True
    assert sliced.base is not None


def test_fancy_indexing_returns_a_copy() -> None:
    x_grid, _ = create_spatiotemporal_grid(10, 10)
    indices = np.array([0, 2, 4], dtype=np.intp)
    selected, is_view = fancy_index_is_copy(x_grid, indices)
    assert is_view is False
    assert selected.base is None


# --- Module 4.2 : ingestion Polars Lazy + broadcasting -----------------------


def test_ingest_sensor_coordinates_filters_invalid_rows() -> None:
    latitude, longitude = ingest_sensor_coordinates(DATA_PATH)
    assert latitude.size > 0
    assert np.all((latitude >= -90.0) & (latitude <= 90.0))
    assert np.all((longitude >= -180.0) & (longitude <= 180.0))


def test_apply_residual_vectorized_uses_broadcasting_without_python_loops() -> None:
    def dummy_residual(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return x + t

    x_coords = np.array([1.0, 2.0, 3.0])
    t_coords = np.array([10.0, 20.0])
    result = apply_residual_vectorized(dummy_residual, x_coords, t_coords)

    assert result.shape == (3, 2)
    assert np.allclose(result, x_coords.reshape(-1, 1) + t_coords.reshape(1, -1))


# --- Module 5 : stabilite, conditionnement, precision ------------------------


def test_build_hilbert_matrix_is_symmetric_and_ill_conditioned() -> None:
    matrix = build_hilbert_matrix(8)
    assert np.allclose(matrix, matrix.T)
    assert condition_number(matrix) > 1e6


@pytest.mark.parametrize("n", [5, 10, 15])
def test_condition_number_increases_with_dimension(n: int) -> None:
    small = condition_number(build_hilbert_matrix(5))
    larger = condition_number(build_hilbert_matrix(n))
    if n > 5:
        assert larger >= small


def test_reconstruction_error_by_precision_orders_float_precisions() -> None:
    matrix = build_hilbert_matrix(6)
    rhs = np.ones(6)
    errors = reconstruction_error_by_precision(matrix, rhs)
    assert set(errors.keys()) == {"float16", "float32", "float64"}
    assert all(value >= 0.0 for value in errors.values())


def test_perturb_and_propagate_amplifies_error_with_condition_number() -> None:
    matrix = build_hilbert_matrix(10)
    rhs = np.ones(10)
    _, _, relative_error = perturb_and_propagate(matrix, rhs, epsilon=1e-7)
    assert relative_error >= 0.0
    assert np.isfinite(relative_error)


def test_compute_residual_and_validate_solution_with_isclose() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    alpha = np.linalg.solve(matrix, rhs)

    residual = compute_residual(matrix, alpha, rhs)
    assert residual < 1e-10
    assert validate_solution(matrix, alpha, rhs, atol=1e-8, rtol=1e-8)


def test_validate_solution_rejects_wrong_solution() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    wrong_alpha = np.array([100.0, 100.0])
    assert not validate_solution(matrix, wrong_alpha, rhs, atol=1e-6, rtol=1e-6)
