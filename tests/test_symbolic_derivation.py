"""Tests unitaires du Module 3 : derivation symbolique (SymPy)."""

from __future__ import annotations

import numpy as np
import sympy as sp
from src.symbolic_derivation import (
    build_residual_expression,
    compute_derivatives,
    get_symbols,
    lambdify_solution,
    residual_source_term,
    solitary_wave_solution,
)


def test_get_symbols_returns_four_real_symbols() -> None:
    x, t, c, nu = get_symbols()
    assert {x.name, t.name, c.name, nu.name} == {"x", "t", "c", "nu"}
    assert all(symbol.is_real for symbol in (x, t, c, nu))


def test_solitary_wave_solution_is_tanh() -> None:
    x, t, c, _ = get_symbols()
    u = solitary_wave_solution(x, t, c)
    assert sp.simplify(u - sp.tanh(x - c * t)) == 0
    assert any(atom.func == sp.tanh for atom in u.atoms(sp.tanh))


def test_compute_derivatives_matches_analytical_tanh_derivative() -> None:
    x, t, c, _ = get_symbols()
    u = solitary_wave_solution(x, t, c)
    du_dt, du_dx, d2u_dx2 = compute_derivatives(u, x, t)

    # d/dx tanh(x - c t) = 1 - tanh(x - c t)^2 ; d/dt tanh(x - c t) = -c (1 - tanh^2)
    expected_du_dx = sp.simplify(1 - sp.tanh(x - c * t) ** 2)
    assert sp.simplify(du_dx - expected_du_dx) == 0
    assert sp.simplify(du_dt + c * du_dx) == 0
    assert d2u_dx2 != 0


def test_residual_source_term_is_symbolic_expression() -> None:
    x, t, c, nu = get_symbols()
    u = solitary_wave_solution(x, t, c)
    du_dt, du_dx, d2u_dx2 = compute_derivatives(u, x, t)
    f = residual_source_term(du_dt, du_dx, d2u_dx2, c, nu)
    assert isinstance(f, sp.Expr)
    assert nu in f.free_symbols


def test_build_residual_expression_substitutes_numeric_parameters() -> None:
    x, t, u_num, f_num = build_residual_expression(c_value=1.0, nu_value=0.05)
    remaining_symbols = f_num.free_symbols
    assert sp.Symbol("c", real=True) not in remaining_symbols
    assert sp.Symbol("nu", real=True) not in remaining_symbols
    assert x in u_num.free_symbols or t in u_num.free_symbols


def test_lambdify_solution_produces_vectorized_numpy_callables() -> None:
    u_func, f_func = lambdify_solution(c_value=1.0, nu_value=0.05)
    x_values = np.linspace(-2.0, 2.0, 10)
    t_values = np.zeros_like(x_values)

    u_values = u_func(x_values, t_values)
    f_values = f_func(x_values, t_values)

    assert u_values.shape == x_values.shape
    assert np.allclose(u_values, np.tanh(x_values))
    assert np.all(np.isfinite(f_values))
