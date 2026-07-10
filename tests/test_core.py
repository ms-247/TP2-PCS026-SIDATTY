"""Suite de tests d'integration : execute le pipeline complet de bout en bout.

Valide l'enchainement derivation symbolique -> grille -> entrainement PINN
(rapide) -> generation des figures, sur un tres petit probleme afin de
rester compatible avec les contraintes de temps d'un runner CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.generate_artefacts import build_reference_grid, generate_plots, train_and_save
from src.visualization import plot_prediction_and_error


def test_build_reference_grid_matches_symbolic_solution() -> None:
    x_grid, t_grid, u_exact = build_reference_grid(n_x=10, n_t=10)
    assert x_grid.shape == (10, 10)
    assert u_exact.shape == (10, 10)


def test_build_reference_grid_returns_ndarrays_at_minimal_grid_size(tmp_path: Path) -> None:
    """Regression test (exercice 9.2) : les grilles issues du pipeline doivent
    rester des np.ndarray, meme au cas limite n_x=n_t=2, pour pouvoir etre
    passees telles quelles a plot_prediction_and_error (une chaine de
    caracteres avait ete injectee par erreur, bloquee par mypy --strict).
    """
    x_grid, t_grid, u_exact = build_reference_grid(n_x=2, n_t=2)

    assert isinstance(x_grid, np.ndarray)
    assert isinstance(t_grid, np.ndarray)
    assert isinstance(u_exact, np.ndarray)

    absolute_error = np.abs(u_exact - u_exact)
    output_path = plot_prediction_and_error(
        x_grid, t_grid, u_exact, absolute_error, tmp_path / "prediction_error.pdf", use_tex=False
    )
    assert output_path.exists()


def test_end_to_end_pipeline_produces_all_artefacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import generate_artefacts

    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(generate_artefacts, "OUTPUT_DIR", output_dir)

    model_path = output_dir / "models" / "pinn_weights.pt"
    train_and_save(model_path, epochs=5)
    assert model_path.exists()

    pdf_path, html_path = generate_plots(model_path)
    assert pdf_path.exists()
    assert html_path.exists()
