"""Time-domain weak-form assembly utilities for ODE identification."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson


def make_test_basis(time, num_basis):
    """Build globally supported sinusoidal test functions and derivatives."""
    time = np.asarray(time, dtype=float)
    t_min, t_max = time[0], time[-1]
    t_range = t_max - t_min

    basis = []

    for k in range(1, num_basis + 1):
        # Frequency
        omega = k * np.pi / t_range

        # Sinusoidal basis function
        phi = np.sin(omega * (time - t_min))

        # Derivative
        dphi = omega * np.cos(omega * (time - t_min))

        basis.append((phi, dphi))

    return basis


def weak_library_rows(states, time, test_basis, library_fn, component_index):
    """Assemble one-component weak-form rows for first-order systems."""
    library, _ = library_fn(states)
    component = states[:, component_index]

    A_rows = []
    b_rows = []

    for phi, dphi in test_basis:
        A_rows.append(simpson(library * phi[:, None], x=time, axis=0))
        b_rows.append(-simpson(component * dphi, x=time))

    return np.vstack(A_rows), np.asarray(b_rows)


def build_weak_system(trajectories, time, test_basis, library_fn, component_index):
    """Stack weak-form rows over trajectories for one first-order component."""
    A_blocks = []
    b_blocks = []

    for states in trajectories:
        A_k, b_k = weak_library_rows(
            states, time, test_basis, library_fn, component_index
        )
        A_blocks.append(A_k)
        b_blocks.append(b_k)

    return np.vstack(A_blocks), np.concatenate(b_blocks)


def weak_rows_second_order(states, time, test_basis, library_fn, forcing_fn):
    """Assemble weak-form rows for one second-order scalar equation."""
    library, _ = library_fn(states)
    v = states[:, 1]
    forcing = np.asarray(forcing_fn(time), dtype=float)

    A_rows = []
    b_rows = []
    for phi, dphi in test_basis:
        deriv_col = -simpson(v * dphi, x=time)
        alg_cols = simpson(library * phi[:, None], x=time, axis=0)
        A_rows.append(np.concatenate([[deriv_col], alg_cols]))
        b_rows.append(simpson(forcing * phi, x=time))

    return np.vstack(A_rows), np.asarray(b_rows)


def build_weak_system_second_order(trajectories, time, test_basis, library_fn, forcing_fn):
    """Stack weak-form rows over trajectories for second-order systems."""
    A_blocks = []
    b_blocks = []

    for states in trajectories:
        A_k, b_k = weak_rows_second_order(
            states, time, test_basis, library_fn, forcing_fn
        )
        A_blocks.append(A_k)
        b_blocks.append(b_k)
    return np.vstack(A_blocks), np.concatenate(b_blocks)
