"""Generic ODE solver using learned or specified coefficients and library functions."""

import numpy as np
from scipy.integrate import solve_ivp


def solve_ode(library_func, coefficients, initial_condition, time, method="RK45", rtol=1e-10, atol=1e-12):
    """
    Integrate an ODE system defined by library functions and coefficients.
    
    The ODE is defined as:
        d(state)/dt = coefficients · library_func(state)
    
    For a system with n components (e.g., 3D Lorenz), coefficients should be
    a dict with keys for each component, each containing a coefficient vector.
    
    Args:
        library_func: Callable that takes states array (N, d) and returns (N, p) library matrix
                     and list of p term names. E.g., lorenz_library(states)
        coefficients: Either:
                     - dict with component keys (e.g., "x", "y", "z") mapping to coefficient vectors
                     - single coefficient vector for 1D systems
        initial_condition: Initial state (array-like)
        time: Time array for integration (t_eval parameter)
        method: Integration method for solve_ivp (default: "RK45")
        rtol: Relative tolerance (default: 1e-10)
        atol: Absolute tolerance (default: 1e-12)
    
    Returns:
        trajectory: (N, d) array where N = len(time), d = dimension of state space
    
    Raises:
        RuntimeError: If integration fails
    """
    
    def rhs(_, state):
        """Right-hand side of ODE."""
        # Evaluate library at current state
        library_state = np.array([state]).reshape(1, -1)
        lib_vals, _ = library_func(library_state)
        lib_vals = lib_vals[0]  # Extract single evaluation
        
        # Handle both dict and vector coefficient formats
        if isinstance(coefficients, dict):
            # Multi-component system: each component has its own coefficients
            derivatives = np.array([
                np.dot(coefficients[key], lib_vals)
                for key in sorted(coefficients.keys())
            ])
        else:
            # Single component: scalar or 1D array
            derivatives = np.dot(coefficients, lib_vals)
        
        return derivatives
    
    solution = solve_ivp(
        rhs,
        (float(time[0]), float(time[-1])),
        np.asarray(initial_condition, dtype=float),
        t_eval=time,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    
    return solution.y.T
