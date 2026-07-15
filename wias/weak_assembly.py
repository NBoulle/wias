"""Generic weak-form library assembly using UFL expressions."""

from __future__ import annotations

import numpy as np
from collections.abc import Mapping

try:
    import firedrake as fd
except ImportError as exc:
    raise ImportError("Firedrake is required for weak_assembly.") from exc


def assemble_weak_library_rows(library_expressions, rhs, bcs=None):
    """
    Assemble weak-form library rows given UFL expressions.

    Parameters
    ----------

    library_expressions : dict[str, UFL expression] or iterable[UFL expression]
        Weak-form library terms. If a dict is provided, terms are assembled in
        insertion order of the dictionary values.
    rhs : RHS in weak form

    bcs : firedrake.DirichletBC or list, optional
        Boundary conditions to apply during assembly. If None, no BCs.

    Returns
    -------
    A : ndarray, shape (n_dofs, n_library)
        Weak-form matrix where column j is ∫library[j] * v_i dx.
    b : ndarray, shape (n_dofs,)
        RHS vector: ∫f * v_i dx.
    """

    # Support both dictionary-based and iterable libraries.
    if isinstance(library_expressions, Mapping):
        expressions = list(library_expressions.values())
    else:
        expressions = list(library_expressions)

    # Assemble each library term
    columns = []
    for expr in expressions:
        col_vector = fd.assemble(expr, bcs=bcs)
        with col_vector.dat.vec_ro as cv:
            col_array = cv.getArray(readonly=True).copy()
        columns.append(col_array)

    # Assemble RHS
    rhs_vector = fd.assemble(rhs, bcs=bcs)
    with rhs_vector.dat.vec_ro as rv:
        b = rv.getArray(readonly=True).copy()

    # Stack columns
    A = np.column_stack(columns)

    return A, b
