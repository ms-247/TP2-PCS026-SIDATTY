"""Visualisation scientifique pour publication (Module 8).

Genere une figure statique double-panneaux (heatmap PINN + erreur temporelle)
au format vectoriel PDF avec rendu LaTeX natif, ainsi qu'une surface 3D
interactive u_hat(x, t) exportee en HTML autonome via Plotly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import seaborn as sns

FloatArray = npt.NDArray[np.float64]


def configure_latex_rendering(use_tex: bool = True) -> None:
    """Force l'integration native de LaTeX pour le rendu des labels mathematiques."""
    matplotlib.rcParams["text.usetex"] = use_tex
    matplotlib.rcParams["font.family"] = "serif"


def plot_prediction_and_error(
    x_grid: FloatArray,
    t_grid: FloatArray,
    u_predicted: FloatArray,
    absolute_error: FloatArray,
    output_path: Path,
    use_tex: bool = True,
) -> Path:
    """Figure double-panneaux : heatmap de u_hat(x,t) et evolution temporelle de l'erreur.

    Exporte exclusivement au format vectoriel .pdf, conforme aux exigences
    de publication scientifique. use_tex=False permet de desactiver le rendu
    LaTeX natif sur les environnements sans installation TeX (ex. runners CI).
    """
    configure_latex_rendering(use_tex)
    sns.set_theme(style="whitegrid")
    fig, (ax_heatmap, ax_error) = plt.subplots(1, 2, figsize=(11, 4.5))

    mesh = ax_heatmap.pcolormesh(t_grid, x_grid, u_predicted, shading="auto", cmap="viridis")
    ax_heatmap.set_xlabel(r"$t$")
    ax_heatmap.set_ylabel(r"$x$")
    ax_heatmap.set_title(r"Solution approchee $\hat{u}(x, t)$ (PINN)")
    fig.colorbar(mesh, ax=ax_heatmap)

    mean_error_over_time = np.mean(absolute_error, axis=0)
    time_axis = t_grid[0, :]
    ax_error.plot(time_axis, mean_error_over_time, color="firebrick")
    ax_error.set_xlabel(r"$t$")
    ax_error.set_ylabel(r"Erreur absolue moyenne $|\hat{u} - u|$")
    ax_error.set_title("Evolution temporelle de l'erreur")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    return output_path


def plot_interactive_surface(
    x_grid: FloatArray, t_grid: FloatArray, u_predicted: FloatArray, output_path: Path
) -> Path:
    """Surface 3D interactive u_hat(x,t) exportee en fichier HTML autonome (Plotly)."""
    figure = go.Figure(
        data=[go.Surface(x=t_grid, y=x_grid, z=u_predicted, colorscale="Viridis")]
    )
    figure.update_layout(
        title="Surface interactive u_hat(x, t)",
        scene={"xaxis_title": "t", "yaxis_title": "x", "zaxis_title": "u_hat"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    return output_path
