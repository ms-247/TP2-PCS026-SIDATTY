"""Tests unitaires du Module 8 : visualisation scientifique (Matplotlib/Plotly)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.numerical_core import create_spatiotemporal_grid
from src.visualization import plot_interactive_surface, plot_prediction_and_error


def test_plot_prediction_and_error_creates_pdf(tmp_path: Path) -> None:
    x_grid, t_grid = create_spatiotemporal_grid(10, 10)
    u_predicted = np.tanh(x_grid - t_grid)
    absolute_error = np.abs(u_predicted - np.tanh(x_grid))

    output_path = tmp_path / "figures" / "prediction_error.pdf"
    result_path = plot_prediction_and_error(
        x_grid, t_grid, u_predicted, absolute_error, output_path, use_tex=False
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_interactive_surface_creates_html(tmp_path: Path) -> None:
    x_grid, t_grid = create_spatiotemporal_grid(8, 8)
    u_predicted = np.tanh(x_grid - t_grid)

    output_path = tmp_path / "figures" / "surface_3d.html"
    result_path = plot_interactive_surface(x_grid, t_grid, u_predicted, output_path)

    assert result_path == output_path
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "<html" in content.lower()
