"""Derivation symbolique de l'equation d'advection-diffusion-reaction (SymPy).

Solution candidate de type onde solitaire u(x, t) = tanh(x - c*t), injectee
dans l'equation d'advection-diffusion pour generer le terme source residuel
exact f(x, t) = du/dt + c*du/dx - nu*d2u/dx2, puis compilee en fonctions
NumPy vectorisees via sympy.lambdify.
"""

from __future__ import annotations

from typing import Callable, cast

import numpy as np
import numpy.typing as npt
import sympy as sp

FloatArray = npt.NDArray[np.float64]


def get_symbols() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]:
    """Retourne les symboles (x, t, c, nu) : espace, temps, advection, diffusion."""
    x, t, c, nu = sp.symbols("x t c nu", real=True)
    return x, t, c, nu


def solitary_wave_solution(x: sp.Symbol, t: sp.Symbol, c: sp.Symbol) -> sp.Expr:
    """Solution analytique candidate : onde solitaire u(x, t) = tanh(x - c*t)."""
    return cast(sp.Expr, sp.tanh(x - c * t))


def compute_derivatives(
    u: sp.Expr, x: sp.Symbol, t: sp.Symbol
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Calcule (du/dt, du/dx, d2u/dx2) par derivation symbolique exacte."""
    du_dt = cast(sp.Expr, sp.diff(u, t))
    du_dx = cast(sp.Expr, sp.diff(u, x))
    d2u_dx2 = cast(sp.Expr, sp.diff(u, x, 2))
    return du_dt, du_dx, d2u_dx2


def residual_source_term(
    du_dt: sp.Expr, du_dx: sp.Expr, d2u_dx2: sp.Expr, c: sp.Symbol, nu: sp.Symbol
) -> sp.Expr:
    """Terme source residuel exact f(x,t) = du/dt + c*du/dx - nu*d2u/dx2."""
    f = du_dt + c * du_dx - nu * d2u_dx2
    return cast(sp.Expr, sp.simplify(f))


def build_residual_expression(
    c_value: float, nu_value: float
) -> tuple[sp.Symbol, sp.Symbol, sp.Expr, sp.Expr]:
    """Construit u(x,t) et f(x,t) numeriquement parametres par c et nu.

    Retourne (x, t, u_expr, f_expr) avec c et nu deja substitues par leurs
    valeurs numeriques, pretes pour lambdify.
    """
    x, t, c, nu = get_symbols()
    u = solitary_wave_solution(x, t, c)
    du_dt, du_dx, d2u_dx2 = compute_derivatives(u, x, t)
    f = residual_source_term(du_dt, du_dx, d2u_dx2, c, nu)

    substitutions = {c: c_value, nu: nu_value}
    u_num = cast(sp.Expr, u.subs(substitutions))
    f_num = cast(sp.Expr, f.subs(substitutions))
    return x, t, u_num, f_num


def lambdify_solution(
    c_value: float, nu_value: float
) -> tuple[Callable[[FloatArray, FloatArray], FloatArray], Callable[[FloatArray, FloatArray], FloatArray]]:
    """Compile u(x,t) et f(x,t) en fonctions NumPy vectorisees via lambdify.

    Retourne un couple (u_func, f_func) applicables directement sur des
    tableaux NumPy (x, t) de meme forme.
    """
    x, t, u_num, f_num = build_residual_expression(c_value, nu_value)

    u_func = cast(
        Callable[[FloatArray, FloatArray], FloatArray],
        sp.lambdify((x, t), u_num, modules="numpy"),
    )
    f_func = cast(
        Callable[[FloatArray, FloatArray], FloatArray],
        sp.lambdify((x, t), f_num, modules="numpy"),
    )
    return u_func, f_func
