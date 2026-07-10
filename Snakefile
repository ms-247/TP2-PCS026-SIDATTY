"""Orchestration Snakemake du pipeline scientifique complet (Module 9).

Graphe de dependances acyclique dirige (DAG) :
    derive_symbolic -> ingest_and_vectorize -> analyze_stability -> train_pinn -> generate_plots

Execution complete : `snakemake --cores 1 all`
"""

OUTPUTS = "outputs"
DATA = "data/raw_sensors"


rule all:
    input:
        f"{OUTPUTS}/figures/prediction_error.pdf",
        f"{OUTPUTS}/figures/surface_3d.html",
        f"{OUTPUTS}/models/pinn_weights.pt",
        f"{OUTPUTS}/symbolic/residual.txt",
        f"{OUTPUTS}/stability/report.txt",
        f"{OUTPUTS}/vectorized/coordinates.npz",


rule derive_symbolic:
    """Genere les fonctions residuelles lambdifiees (SymPy -> NumPy)."""
    output:
        f"{OUTPUTS}/symbolic/residual.txt",
    run:
        from pathlib import Path

        from src.symbolic_derivation import build_residual_expression

        _, _, u_expr, f_expr = build_residual_expression(c_value=1.0, nu_value=0.05)
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(output[0]).write_text(f"u(x,t) = {u_expr}\nf(x,t) = {f_expr}\n")


rule ingest_and_vectorize:
    """Filtre les donnees Parquet et prepare les matrices de discretisation."""
    input:
        sensors=f"{DATA}/sensors.parquet",
    output:
        f"{OUTPUTS}/vectorized/coordinates.npz",
    run:
        from pathlib import Path

        import numpy as np

        from src.numerical_core import ingest_sensor_coordinates

        latitude, longitude = ingest_sensor_coordinates(Path(input.sensors))
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        np.savez(output[0], latitude=latitude, longitude=longitude)


rule analyze_stability:
    """Evalue le conditionnement et la tolerance IEEE 754 (matrice de Hilbert)."""
    output:
        f"{OUTPUTS}/stability/report.txt",
    run:
        from pathlib import Path

        from src.numerical_core import build_hilbert_matrix, condition_number

        lines = []
        for n in range(5, 26, 5):
            matrix = build_hilbert_matrix(n)
            lines.append(f"n={n}: kappa(A)={condition_number(matrix):.4e}")
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        Path(output[0]).write_text("\n".join(lines) + "\n")


rule train_pinn:
    """Entraine le reseau PINN PyTorch et sauvegarde les poids du modele."""
    input:
        f"{OUTPUTS}/symbolic/residual.txt",
    output:
        f"{OUTPUTS}/models/pinn_weights.pt",
    run:
        from pathlib import Path

        from src.generate_artefacts import train_and_save

        train_and_save(Path(output[0]))


rule generate_plots:
    """Produit les rapports visuels statiques (PDF) et interactifs (HTML)."""
    input:
        model=f"{OUTPUTS}/models/pinn_weights.pt",
    output:
        f"{OUTPUTS}/figures/prediction_error.pdf",
        f"{OUTPUTS}/figures/surface_3d.html",
    run:
        from pathlib import Path

        from src.generate_artefacts import generate_plots as generate_plots_fn

        generate_plots_fn(Path(input.model))
