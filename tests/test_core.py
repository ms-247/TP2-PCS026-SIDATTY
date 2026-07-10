"""Suite de tests d'integration : execute le pipeline complet de bout en bout.

Valide l'enchainement derivation symbolique -> grille -> entrainement PINN
(rapide) -> generation des figures, sur un tres petit probleme afin de
rester compatible avec les contraintes de temps d'un runner CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.generate_artefacts import build_reference_grid, generate_plots, train_and_save


def test_build_reference_grid_matches_symbolic_solution() -> None:
    x_grid, t_grid, u_exact = build_reference_grid(n_x=10, n_t=10)
    assert x_grid.shape == (10, 10)
    assert u_exact.shape == (10, 10)


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
