"""Top-level package exports for wias."""

from .linear_algebra import (
    recover_homogeneous_relation,
    CPQR_solver,
)
from .ode import solve_ode
from .weak_form import (
    build_weak_system,
    build_weak_system_second_order,
    make_test_basis,
    weak_library_rows,
    weak_rows_second_order,
)

try:
    from .weak_assembly import (
        assemble_weak_library_rows,
    )
    _has_firedrake = True
except ImportError:
    _has_firedrake = False

__all__ = [
    "build_weak_system",
    "build_weak_system_second_order",
    "make_test_basis",
    "recover_homogeneous_relation",
    "CPQR_solver",
    "solve_ode",
    "weak_library_rows",
    "weak_rows_second_order",
    "assemble_weak_library_rows",
]
