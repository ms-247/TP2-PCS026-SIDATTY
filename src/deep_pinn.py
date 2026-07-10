"""Reseau de neurones informe par la physique (Physics-Informed Neural Network).

Module 7 : un perceptron multicouche approxime u_hat(x, t), entraine par
retropropagation d'une perte composite melangeant le residu physique de
l'equation d'advection-diffusion (calcule par differentiation automatique)
et l'erreur aux donnees/conditions aux limites. Le code detecte et exploite
automatiquement l'accelerateur materiel disponible (CPU, CUDA, MPS).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


def get_device() -> torch.device:
    """Detecte et retourne le meilleur accelerateur materiel disponible.

    Ordre de priorite : CUDA (NVIDIA) > MPS (Apple Silicon) > CPU. Cette
    abstraction permet a l'entrainement de s'executer de maniere identique
    quelle que soit l'architecture, prefigurant un passage a l'echelle
    multi-GPU (torch.nn.DataParallel / DistributedDataParallel) sans
    modification du code applicatif.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PINN(nn.Module):
    """MLP prenant (x, t) en entree et retournant l'approximation u_hat(x, t)."""

    def __init__(self, hidden_layers: int = 3, hidden_units: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.Tanh()]
        layers.append(nn.Linear(hidden_units, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        inputs = torch.cat([x, t], dim=1)
        output = self.network(inputs)
        return cast(Tensor, output)


def physics_residual_loss(model: PINN, x: Tensor, t: Tensor, c: float, nu: float) -> Tensor:
    """Perte physique sans maillage : residu de l'EDP calcule par autograd.

    Exploite torch.autograd.grad pour obtenir les operateurs differentiels
    d'ordre superieur (du/dt, du/dx, d2u/dx2) directement au point (x, t),
    sans discretisation spatiale explicite.
    """
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)

    grad_outputs = torch.ones_like(u)
    du_dx = cast(
        Tensor, torch.autograd.grad(u, x, grad_outputs=grad_outputs, create_graph=True)[0]
    )
    du_dt = cast(
        Tensor, torch.autograd.grad(u, t, grad_outputs=grad_outputs, create_graph=True)[0]
    )
    d2u_dx2 = cast(
        Tensor,
        torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0],
    )

    residual = du_dt + c * du_dx - nu * d2u_dx2
    return cast(Tensor, torch.mean(residual**2))


def data_loss(model: PINN, x_data: Tensor, t_data: Tensor, u_data: Tensor) -> Tensor:
    """Erreur quadratique moyenne aux points de mesure issus des capteurs."""
    prediction = model(x_data, t_data)
    return cast(Tensor, torch.mean((prediction - u_data) ** 2))


def total_loss(
    model: PINN,
    x_physics: Tensor,
    t_physics: Tensor,
    x_data: Tensor,
    t_data: Tensor,
    u_data: Tensor,
    c: float,
    nu: float,
    physics_weight: float = 1.0,
    data_weight: float = 1.0,
) -> Tensor:
    """Combine perte physique et perte donnees/conditions aux limites."""
    l_physics = physics_residual_loss(model, x_physics, t_physics, c, nu)
    l_data = data_loss(model, x_data, t_data, u_data)
    return cast(Tensor, physics_weight * l_physics + data_weight * l_data)


def train_pinn(
    model: PINN,
    x_physics: Tensor,
    t_physics: Tensor,
    x_data: Tensor,
    t_data: Tensor,
    u_data: Tensor,
    c: float,
    nu: float,
    epochs: int = 200,
    lr: float = 1e-3,
) -> list[float]:
    """Entraine le PINN par descente de gradient Adam et retourne l'historique de perte."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = total_loss(model, x_physics, t_physics, x_data, t_data, u_data, c, nu)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    return history
