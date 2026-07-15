import numpy as np
import wias


def _affine_library(states):
    x = states[:, 0]
    return np.column_stack([np.ones_like(x), x]), ["1", "x"]


def test_import_and_simple_dictionary_fit():
    # Verify import provides core weak-SINDy building blocks.
    assert hasattr(wias, "build_weak_system")
    assert hasattr(wias, "make_test_basis")
    assert hasattr(wias, "CPQR_solver")

    # Exact trajectory for x' = 2x - 2: x(t) = 1 + exp(2t).
    time = np.linspace(0.0, 1.0, 400)
    x = 1.0 + np.exp(2.0 * time)
    states = x[:, None]

    test_basis = wias.make_test_basis(time, num_basis=25)
    A, b = wias.build_weak_system(
        trajectories=[states],
        time=time,
        test_basis=test_basis,
        library_fn=_affine_library,
        component_index=0,
    )

    coeffs = wias.CPQR_solver(A, b)

    np.testing.assert_allclose(coeffs, np.array([-2.0, 2.0]), atol=7e-2)
