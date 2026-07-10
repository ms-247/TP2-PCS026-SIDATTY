"""Tests unitaires du Module 7 : PINN PyTorch, autograd et device abstraction."""

from __future__ import annotations

import torch

from src.deep_pinn import PINN, data_loss, get_device, physics_residual_loss, total_loss, train_pinn


def test_get_device_returns_a_valid_torch_device() -> None:
    device = get_device()
    assert device.type in {"cuda", "mps", "cpu"}


def test_pinn_forward_shape() -> None:
    model = PINN(hidden_layers=2, hidden_units=8)
    x = torch.rand(5, 1)
    t = torch.rand(5, 1)
    output = model(x, t)
    assert output.shape == (5, 1)


def test_physics_residual_loss_is_scalar_and_differentiable() -> None:
    model = PINN(hidden_layers=2, hidden_units=8)
    x = torch.rand(16, 1)
    t = torch.rand(16, 1)

    loss = physics_residual_loss(model, x, t, c=1.0, nu=0.05)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_data_loss_is_zero_for_perfect_prediction() -> None:
    model = PINN(hidden_layers=1, hidden_units=4)
    x = torch.zeros(3, 1)
    t = torch.zeros(3, 1)
    with torch.no_grad():
        perfect_prediction = model(x, t)

    loss = data_loss(model, x, t, perfect_prediction)
    assert loss.item() == 0.0


def test_total_loss_combines_physics_and_data_terms() -> None:
    model = PINN(hidden_layers=2, hidden_units=8)
    x = torch.rand(8, 1)
    t = torch.rand(8, 1)
    u_data = torch.rand(8, 1)

    loss = total_loss(model, x, t, x, t, u_data, c=1.0, nu=0.05)
    assert torch.isfinite(loss)


def test_train_pinn_reduces_loss_over_epochs() -> None:
    torch.manual_seed(0)
    model = PINN(hidden_layers=2, hidden_units=8)
    x = torch.linspace(-1, 1, 20).reshape(-1, 1)
    t = torch.zeros(20, 1)
    u_data = torch.tanh(x)

    history = train_pinn(model, x, t, x, t, u_data, c=1.0, nu=0.05, epochs=30, lr=1e-2)

    assert len(history) == 30
    assert history[-1] <= history[0]
