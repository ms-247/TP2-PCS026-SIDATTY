"""Script d'orchestration : execute le pipeline complet et produit les artefacts.

Point d'entree utilise par le job CI/CD (upload-artifact) et par les regles
Snakemake `train_pinn` / `generate_plots`. Genere les figures dans
outputs/figures/ et le modele PINN entraine dans outputs/models/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from src.deep_pinn import PINN, get_device, train_pinn
from src.numerical_core import create_spatiotemporal_grid
from src.symbolic_derivation import lambdify_solution
from src.visualization import plot_interactive_surface, plot_prediction_and_error

OUTPUT_DIR = Path("outputs")
MODEL_PATH = OUTPUT_DIR / "models" / "pinn_weights.pt"

FloatArray = npt.NDArray[np.float64]


def build_reference_grid(n_x: int = 40, n_t: int = 40) -> tuple[FloatArray, FloatArray, FloatArray]:
    c_value, nu_value = 1.0, 0.05
    u_func, _ = lambdify_solution(c_value, nu_value)
    x_grid, t_grid = create_spatiotemporal_grid(n_x, n_t)
    u_exact = u_func(x_grid, t_grid)
    return x_grid, t_grid, u_exact


def train_and_save(model_path: Path = MODEL_PATH, epochs: int = 50) -> Path:
    """Entraine le PINN sur la solution de reference et sauvegarde les poids."""
    c_value, nu_value = 1.0, 0.05
    x_grid, t_grid, u_exact = build_reference_grid()

    device = get_device()
    model = PINN(hidden_layers=3, hidden_units=16).to(device)

    x_flat = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32, device=device)
    t_flat = torch.tensor(t_grid.reshape(-1, 1), dtype=torch.float32, device=device)
    u_flat = torch.tensor(u_exact.reshape(-1, 1), dtype=torch.float32, device=device)

    train_pinn(
        model,
        x_flat,
        t_flat,
        x_flat[:50],
        t_flat[:50],
        u_flat[:50],
        c_value,
        nu_value,
        epochs=epochs,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model_path


def generate_plots(model_path: Path = MODEL_PATH) -> tuple[Path, Path]:
    """Recharge le modele entraine et produit les figures statiques/interactives."""
    x_grid, t_grid, u_exact = build_reference_grid()

    device = get_device()
    model = PINN(hidden_layers=3, hidden_units=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x_flat = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32, device=device)
    t_flat = torch.tensor(t_grid.reshape(-1, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        u_predicted = model(x_flat, t_flat).cpu().numpy().reshape(x_grid.shape)

    absolute_error = np.abs(u_predicted - u_exact)

    pdf_path = plot_prediction_and_error(
        "invalid_x_grid_type",
        t_grid,
        u_predicted,
        absolute_error,
        OUTPUT_DIR / "figures" / "prediction_error.pdf",
        use_tex=False,
    )
    html_path = plot_interactive_surface(
        x_grid, t_grid, u_predicted, OUTPUT_DIR / "figures" / "surface_3d.html"
    )
    return pdf_path, html_path


def main() -> None:
    model_path = train_and_save()
    generate_plots(model_path)


if __name__ == "__main__":
    main()
