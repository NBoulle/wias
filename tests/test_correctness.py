"""Correctness tests for all wias modules.

Covers:
  - linear_algebra: remove_dependent_columns, solve_reduced_system, CPQR_solver,
                    recover_homogeneous_relation
  - ode:            solve_ode (vector and dict coefficient modes)
  - utils:          format_equation
  - weak_form:      make_test_basis, weak_library_rows, build_weak_system,
                    weak_rows_second_order, build_weak_system_second_order
  - end-to-end:     SINDy identification from exact trajectories
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from wias.linear_algebra import (
    CPQR_solver,
    recover_homogeneous_relation,
    remove_dependent_columns,
    solve_reduced_system,
)
from wias.ode import solve_ode
from wias.utils import format_equation
from wias.weak_form import (
    build_weak_system,
    build_weak_system_second_order,
    make_test_basis,
    weak_library_rows,
    weak_rows_second_order,
)


# ---------------------------------------------------------------------------
# Shared library helpers
# ---------------------------------------------------------------------------

def _identity_library(states):
    """Library [x]: single-state identity."""
    return states[:, :1], ["x"]


def _affine_library(states):
    """Library [1, x]."""
    x = states[:, 0]
    return np.column_stack([np.ones_like(x), x]), ["1", "x"]


def _constant_library(states):
    """Library [1]."""
    x = states[:, 0]
    return np.ones((len(x), 1)), ["1"]


def _two_state_library(states):
    """Library [x, y] — identity for 2D systems."""
    return states[:, :2], ["x", "y"]


def _spring_library(states):
    """Library [u, u'] for second-order oscillator."""
    return states[:, :2], ["u", "u'"]


# ===========================================================================
# linear_algebra.remove_dependent_columns
# ===========================================================================

class TestRemoveDependentColumns:

    def test_full_rank_keeps_all_columns(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        A_red, cols = remove_dependent_columns(A)
        assert A_red.shape[1] == 2
        assert set(cols) == {0, 1}

    def test_duplicate_column_removed(self):
        # col 1 = 2 * col 0 → dependent
        A = np.column_stack([np.arange(1.0, 4.0), 2.0 * np.arange(1.0, 4.0), np.array([1.0, 0.0, 1.0])])
        A_red, cols = remove_dependent_columns(A)
        assert A_red.shape[1] == 2

    def test_all_zero_matrix_raises(self):
        with pytest.raises(ValueError):
            remove_dependent_columns(np.zeros((3, 3)))

    def test_single_column_kept(self):
        A = np.array([[1.0], [2.0], [3.0]])
        A_red, cols = remove_dependent_columns(A)
        assert A_red.shape == (3, 1)
        assert list(cols) == [0]

    def test_reduced_matrix_spans_column_space(self):
        """A_red @ x_red = A @ x for any vector in the column space."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((10, 5))
        A[:, 2] = 3.0 * A[:, 0] - A[:, 1]  # make col 2 dependent
        A_red, cols = remove_dependent_columns(A)
        assert A_red.shape[1] == 4

    def test_rtol_parameter_controls_rank(self):
        """A near-zero column is kept with tight rtol and dropped with loose rtol."""
        A = np.eye(3)
        A[:, 2] *= 1e-6  # very small but nonzero
        _, cols_tight = remove_dependent_columns(A, rtol=1e-8)
        _, cols_loose = remove_dependent_columns(A, rtol=1e-4)
        assert 2 in cols_tight   # kept with tight tolerance
        assert 2 not in cols_loose  # dropped with loose tolerance


# ===========================================================================
# linear_algebra.solve_reduced_system
# ===========================================================================

class TestSolveReducedSystem:

    def test_identity_system(self):
        b = np.array([1.0, 2.0, 3.0])
        coeffs = solve_reduced_system(np.eye(3), b, np.array([0, 1, 2]), 3)
        npt.assert_allclose(coeffs, b, atol=1e-12)

    def test_dependent_columns_are_zeroed(self):
        """Columns absent from independent_cols get zero in the full output."""
        A_red = np.eye(2)
        b = np.array([5.0, 7.0])
        # Suppose column index 1 was dependent and was removed
        cols = np.array([0, 2])
        coeffs = solve_reduced_system(A_red, b, cols, 3)
        assert coeffs.shape == (3,)
        assert coeffs[1] == 0.0
        assert coeffs[0] == pytest.approx(5.0)
        assert coeffs[2] == pytest.approx(7.0)

    def test_least_squares_overdetermined(self):
        """Consistent overdetermined system gives exact solution."""
        rng = np.random.default_rng(0)
        A_red = rng.standard_normal((20, 2))
        true_x = np.array([3.0, -1.5])
        b = A_red @ true_x
        cols = np.array([0, 1])
        coeffs = solve_reduced_system(A_red, b, cols, 2)
        npt.assert_allclose(coeffs, true_x, atol=1e-10)


# ===========================================================================
# linear_algebra.CPQR_solver
# ===========================================================================

class TestCPQRSolver:

    def test_square_exact_system(self):
        """2x2 exact system is solved to machine precision."""
        A = np.array([[2.0, -1.0], [1.0, 3.0]])
        true_x = np.array([1.0, -2.0])
        b = A @ true_x
        x = CPQR_solver(A, b)
        npt.assert_allclose(x, true_x, atol=1e-12)

    def test_overdetermined_consistent(self):
        """Consistent overdetermined system recovers exact solution."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 3))
        true_x = np.array([1.0, -2.0, 0.5])
        b = A @ true_x
        x = CPQR_solver(A, b)
        npt.assert_allclose(x, true_x, atol=1e-10)

    def test_dependent_columns_handled(self):
        """Dependent column does not cause failure; residual is small."""
        A = np.array([[1.0, 2.0, 1.0],
                      [2.0, 4.0, 3.0],
                      [0.0, 0.0, 1.0]])
        # col 1 = 2 * col 0
        b = np.array([3.0, 7.0, 1.0])
        x = CPQR_solver(A, b)
        assert np.linalg.norm(A @ x - b) < 1e-10

    def test_output_length_equals_num_columns(self):
        A = np.random.default_rng(1).standard_normal((10, 4))
        b = np.ones(10)
        x = CPQR_solver(A, b)
        assert x.shape == (4,)


# ===========================================================================
# linear_algebra.recover_homogeneous_relation
# ===========================================================================

class TestRecoverHomogeneousRelation:
    """recover_homogeneous_relation expects an overdetermined system (rows > cols)
    where the last right singular vector approximates the null vector."""

    @staticmethod
    def _make_null_matrix(true_coeffs, n_rows=40, seed=5):
        """Build a matrix A whose rows are orthogonal to true_coeffs."""
        rng = np.random.default_rng(seed)
        p = len(true_coeffs)
        B = rng.standard_normal((n_rows, p))
        hat = true_coeffs / np.linalg.norm(true_coeffs)
        # Project out the true_coeffs direction so A @ true_coeffs == 0
        return B - (B @ hat)[:, None] * hat

    def test_recovers_known_null_vector(self):
        """Return vector proportional to the true null vector."""
        true_v = np.array([1.0, -2.0, 1.0])
        A = self._make_null_matrix(true_v)
        coeffs, _, _ = recover_homogeneous_relation(A, normalization_index=0)
        # Normalized at index 0 → coeffs[0] = 1, ratio coeffs[i]/true_v[i] is const
        scale = true_v[0]          # = 1.0 in this case
        npt.assert_allclose(coeffs, true_v / scale, atol=1e-8)

    def test_residual_near_zero_for_consistent_system(self):
        true_v = np.array([1.0, -2.0, 1.0])
        A = self._make_null_matrix(true_v)
        _, _, residual = recover_homogeneous_relation(A)
        assert residual < 1e-8

    def test_normalization_at_index_0(self):
        """coeffs[0] == 1.0 when normalization_index=0."""
        A = self._make_null_matrix(np.array([1.0, -2.0, 1.0]))
        coeffs, _, _ = recover_homogeneous_relation(A, normalization_index=0)
        assert coeffs[0] == pytest.approx(1.0, abs=1e-12)

    def test_normalization_index_none_largest_positive(self):
        """normalization_index=None: coefficient with largest magnitude is positive."""
        A = self._make_null_matrix(np.array([1.0, -2.0, 1.0]))
        coeffs, _, _ = recover_homogeneous_relation(A, normalization_index=None)
        pivot = int(np.argmax(np.abs(coeffs)))
        assert coeffs[pivot] > 0.0

    def test_singular_values_returned_descending(self):
        rng = np.random.default_rng(3)
        A = rng.standard_normal((10, 4))
        _, sv, _ = recover_homogeneous_relation(A, normalization_index=None)
        assert np.all(sv[:-1] >= sv[1:] - 1e-14)

    def test_near_zero_pivot_raises_value_error(self):
        """Normalization at an index with ≈0 coefficient raises ValueError."""
        # Near-null vector is [0, 1, 0]; its first component is ~0
        true_v = np.array([0.0, 1.0, 0.0])
        A = self._make_null_matrix(true_v)
        with pytest.raises(ValueError, match="near-zero"):
            recover_homogeneous_relation(A, normalization_index=0)

    def test_relation_satisfies_homogeneous_equation(self):
        """||A @ coeffs|| should be near zero."""
        true_v = np.array([1.0, -2.0, 1.0])
        A = self._make_null_matrix(true_v)
        coeffs, _, _ = recover_homogeneous_relation(A)
        npt.assert_allclose(A @ coeffs, 0.0, atol=1e-8)


# ===========================================================================
# ode.solve_ode
# ===========================================================================

class TestSolveOde:

    def test_exponential_decay_vector_coeffs(self):
        """dx/dt = -x → x(t) = x0 · e^{-t}."""
        time = np.linspace(0.0, 2.0, 1000)
        x0 = 3.0
        traj = solve_ode(_identity_library, np.array([-1.0]), [x0], time)
        npt.assert_allclose(traj[:, 0], x0 * np.exp(-time), rtol=1e-6)

    def test_affine_ode_vector_coeffs(self):
        """dx/dt = -2 + 3x: exact solution is -2/3 + (x0 + 2/3)*e^{3t}."""
        time = np.linspace(0.0, 0.5, 1000)
        x0 = 1.0
        a, b = -2.0, 3.0
        coeffs = np.array([a, b])
        traj = solve_ode(_affine_library, coeffs, [x0], time)
        expected = -a / b + (x0 + a / b) * np.exp(b * time)
        npt.assert_allclose(traj[:, 0], expected, rtol=1e-5)

    def test_dict_coefficients_decoupled_2d(self):
        """dx/dt = -x, dy/dt = -2y via dict coefficients."""
        time = np.linspace(0.0, 1.0, 500)
        x0, y0 = 2.0, -1.0
        coeffs = {
            "x": np.array([-1.0, 0.0]),  # dx/dt = -1*x + 0*y
            "y": np.array([0.0, -2.0]),  # dy/dt = 0*x - 2*y
        }
        traj = solve_ode(_two_state_library, coeffs, [x0, y0], time)
        npt.assert_allclose(traj[:, 0], x0 * np.exp(-time), rtol=1e-6)
        npt.assert_allclose(traj[:, 1], y0 * np.exp(-2.0 * time), rtol=1e-6)

    def test_output_shape(self):
        time = np.linspace(0.0, 1.0, 200)
        coeffs = {"x": np.array([1.0, 0.0]), "y": np.array([0.0, 1.0])}
        traj = solve_ode(_two_state_library, coeffs, [1.0, 1.0], time)
        assert traj.shape == (200, 2)

    def test_initial_condition_satisfied(self):
        """traj[0] matches the provided initial condition."""
        time = np.linspace(0.0, 1.0, 100)
        x0 = 5.0
        traj = solve_ode(_identity_library, np.array([-1.0]), [x0], time)
        npt.assert_allclose(traj[0, 0], x0, atol=1e-10)

    def test_dict_keys_sorted_alphabetically(self):
        """Dict keys are iterated in sorted order; state dimension must match."""
        time = np.linspace(0.0, 0.5, 100)
        # "x" < "y" alphabetically → component 0 is x, component 1 is y
        coeffs = {"y": np.array([0.0, -1.0]), "x": np.array([-1.0, 0.0])}
        traj = solve_ode(_two_state_library, coeffs, [1.0, 2.0], time)
        assert traj.shape == (100, 2)

    def test_integration_failure_raises_runtime_error(self):
        """A diverging system causes RuntimeError."""
        time = np.linspace(0.0, 1e6, 10)  # extremely long span

        def _exploding(states):
            x = states[:, 0]
            return (x**10)[:, None], ["x^10"]

        with pytest.raises(RuntimeError):
            solve_ode(_exploding, np.array([100.0]), [100.0], time)


# ===========================================================================
# utils.format_equation
# ===========================================================================

class TestFormatEquation:

    def test_basic_output_format(self):
        result = format_equation("x", ["1", "x"], [2.0, -3.5])
        assert result.startswith("x' =")
        assert "2.000000*1" in result
        assert "-3.500000*x" in result

    def test_zero_equation_when_all_below_threshold(self):
        result = format_equation("y", ["1", "x"], [0.005, -0.003], min_magnitude=1e-2)
        assert result == "y' = 0"

    def test_threshold_filters_small_terms(self):
        result = format_equation("x", ["1", "x", "x^2"], [0.001, 5.0, 0.005])
        assert "5.000000*x" in result
        assert "x^2" not in result
        assert "0.001000*1" not in result

    def test_custom_threshold_suppresses_term(self):
        result = format_equation("x", ["x"], [0.5], min_magnitude=1.0)
        assert result == "x' = 0"

    def test_custom_threshold_keeps_term(self):
        result = format_equation("x", ["x"], [2.0], min_magnitude=1.0)
        assert "2.000000*x" in result

    def test_component_name_in_output(self):
        result = format_equation("z", ["1"], [3.0])
        assert result.startswith("z' =")


# ===========================================================================
# weak_form.make_test_basis
# ===========================================================================

class TestMakeTestBasis:

    def setup_method(self):
        self.time = np.linspace(0.0, 1.0, 5001)
        self.n = 10
        self.basis = make_test_basis(self.time, self.n)
        spacing = (self.time[-1] - self.time[0]) / (self.n + 1)
        self.hw = 2.0 * spacing
        self.centers = np.linspace(self.time[0] + self.hw, self.time[-1] - self.hw, self.n)

    def test_correct_number_of_windows(self):
        assert len(self.basis) == self.n

    def test_arrays_have_correct_shape(self):
        for phi, dphi in self.basis:
            assert phi.shape == self.time.shape
            assert dphi.shape == self.time.shape

    def test_phi_nonnegative(self):
        """(1 - s²)² ≥ 0 everywhere."""
        for phi, _ in self.basis:
            assert np.all(phi >= -1e-15)

    def test_phi_unity_at_center(self):
        """phi(center) = (1 - 0²)² = 1."""
        for (phi, _), center in zip(self.basis, self.centers):
            idx = np.argmin(np.abs(self.time - center))
            assert phi[idx] == pytest.approx(1.0, abs=1e-3)

    def test_phi_zero_outside_support(self):
        """phi = 0 strictly outside [center - hw, center + hw]."""
        for (phi, _), center in zip(self.basis, self.centers):
            outside = np.abs(self.time - center) >= self.hw
            npt.assert_array_equal(phi[outside], 0.0)

    def test_dphi_zero_at_center(self):
        """dphi(center) ≈ 0 because dphi = -4s(1-s^2)/hw and s≈0 at the center."""
        # Use a finer grid so the nearest point to each center is very close
        time_fine = np.linspace(0.0, 1.0, 100001)
        basis_fine = make_test_basis(time_fine, self.n)
        spacing = (time_fine[-1] - time_fine[0]) / (self.n + 1)
        hw = 2.0 * spacing
        centers = np.linspace(time_fine[0] + hw, time_fine[-1] - hw, self.n)
        for (_, dphi), center in zip(basis_fine, centers):
            idx = np.argmin(np.abs(time_fine - center))
            assert dphi[idx] == pytest.approx(0.0, abs=1e-2)

    def test_dphi_formula_at_interior_point(self):
        """Verify dphi = -4s(1-s²)/hw at an interior point s ≈ 0.5."""
        # Use num_windows=3 so linspace gives a proper center at the middle window
        time = np.linspace(0.0, 1.0, 100001)
        hw = 0.1
        basis = make_test_basis(time, 3, half_width=hw)
        # Centers: linspace(0+0.1, 1-0.1, 3) = [0.1, 0.5, 0.9]
        _, dphi = basis[1]    # middle window centered at 0.5
        actual_center = np.linspace(0.0 + hw, 1.0 - hw, 3)[1]  # = 0.5
        target = actual_center + 0.5 * hw   # s = 0.5
        idx = np.argmin(np.abs(time - target))
        s = (time[idx] - actual_center) / hw
        expected = -4.0 * s * (1.0 - s**2) / hw
        assert dphi[idx] == pytest.approx(expected, rel=1e-3)

    def test_dphi_zero_outside_support(self):
        """dphi = 0 outside compact support."""
        for (_, dphi), center in zip(self.basis, self.centers):
            outside = np.abs(self.time - center) >= self.hw
            npt.assert_array_equal(dphi[outside], 0.0)

    def test_custom_half_width_respected(self):
        """Explicit half_width overrides the automatic spacing."""
        basis = make_test_basis(self.time, 5, half_width=0.05)
        assert len(basis) == 5
        # Verify support width: half the second-window center distance
        hw_used = 0.05
        phi0, _ = basis[0]
        # Support should vanish at ±hw_used from the center
        # phi with custom hw=0.05 should be zero at indices farther than 0.05
        centers_custom = np.linspace(self.time[0] + hw_used, self.time[-1] - hw_used, 5)
        outside = np.abs(self.time - centers_custom[0]) >= hw_used
        npt.assert_array_equal(phi0[outside], 0.0)


# ===========================================================================
# weak_form.weak_library_rows — integration-by-parts correctness
# ===========================================================================

class TestWeakLibraryRows:

    def setup_method(self):
        self.time = np.linspace(0.0, 1.0, 8001)
        self.basis = make_test_basis(self.time, num_windows=20)

    def test_output_shapes(self):
        states = np.sin(self.time)[:, None]
        A, b, names = weak_library_rows(states, self.time, self.basis, _affine_library, 0)
        assert A.shape == (20, 2)
        assert b.shape == (20,)
        assert names == ["1", "x"]

    def test_ibp_linear_trajectory(self):
        """x(t) = t → dx/dt = 1 = 1·1 + 0·x.  A @ [1, 0] ≈ b via IBP."""
        states = self.time[:, None]
        A, b, names = weak_library_rows(states, self.time, self.basis, _affine_library, 0)
        npt.assert_allclose(A @ np.array([1.0, 0.0]), b, atol=1e-5)

    def test_ibp_exponential_trajectory(self):
        """x(t) = e^t → dx/dt = x = 1·x.  A @ [1] ≈ b via IBP."""
        x = np.exp(self.time)
        states = x[:, None]
        A, b, _ = weak_library_rows(states, self.time, self.basis, _identity_library, 0)
        npt.assert_allclose(A @ np.array([1.0]), b, atol=1e-4)

    def test_ibp_constant_trajectory_gives_zero_rhs(self):
        """x(t) = 1 → dx/dt = 0 → b_j = -∫dphi_j dt ≈ 0 (compact support).

        Note: The integral of dphi over the compact support is analytically 0
        (∫dphi dt = [phi]_boundary = 0), but Simpson's rule introduces a small
        numerical quadrature error on the order of 1e-6.
        """
        states = np.ones((len(self.time), 1))
        _, b, _ = weak_library_rows(states, self.time, self.basis, _constant_library, 0)
        npt.assert_allclose(b, 0.0, atol=2e-5)

    def test_correct_component_index(self):
        """component_index selects the right column of states."""
        # Two-state system: component_index=1 should use states[:,1]
        x = self.time[:, None]
        y = np.exp(self.time)[:, None]
        states = np.hstack([x, y])
        _, b_comp1, _ = weak_library_rows(states, self.time, self.basis, _identity_library, 1)
        # For y(t)=e^t with library [y]: b_j = -∫e^t dphi_j dt ≈ ∫e^t phi_j dt (IBP)
        states_exp = y
        A_ref, b_ref, _ = weak_library_rows(states_exp, self.time, self.basis, _identity_library, 0)
        npt.assert_allclose(b_comp1, b_ref, atol=1e-10)

    def test_names_from_library(self):
        states = np.ones((len(self.time), 1))
        _, _, names = weak_library_rows(states, self.time, self.basis, _affine_library, 0)
        assert names == ["1", "x"]


# ===========================================================================
# weak_form.build_weak_system
# ===========================================================================

class TestBuildWeakSystem:

    def setup_method(self):
        self.time = np.linspace(0.0, 1.0, 4001)
        self.basis = make_test_basis(self.time, num_windows=10)

    def test_single_trajectory_equals_weak_library_rows(self):
        states = np.exp(self.time)[:, None]
        A_ref, b_ref, _ = weak_library_rows(states, self.time, self.basis, _affine_library, 0)
        A, b, _ = build_weak_system([states], self.time, self.basis, _affine_library, 0)
        npt.assert_array_equal(A, A_ref)
        npt.assert_array_equal(b, b_ref)

    def test_two_trajectories_stack_rows(self):
        s1 = np.exp(self.time)[:, None]
        s2 = np.sin(self.time)[:, None]
        A, b, names = build_weak_system([s1, s2], self.time, self.basis, _affine_library, 0)
        assert A.shape == (20, 2)    # 10 windows × 2 trajectories
        assert b.shape == (20,)

    def test_names_consistent_across_trajectories(self):
        s1 = np.exp(self.time)[:, None]
        s2 = np.cos(self.time)[:, None]
        _, _, names = build_weak_system([s1, s2], self.time, self.basis, _affine_library, 0)
        assert names == ["1", "x"]

    def test_row_order_first_trajectory_then_second(self):
        """First 10 rows come from trajectory 1; next 10 from trajectory 2."""
        s1 = np.exp(self.time)[:, None]
        s2 = np.sin(self.time)[:, None]
        A_full, b_full, _ = build_weak_system([s1, s2], self.time, self.basis, _affine_library, 0)
        A1, b1, _ = weak_library_rows(s1, self.time, self.basis, _affine_library, 0)
        A2, b2, _ = weak_library_rows(s2, self.time, self.basis, _affine_library, 0)
        npt.assert_array_equal(A_full[:10], A1)
        npt.assert_array_equal(A_full[10:], A2)
        npt.assert_array_equal(b_full[:10], b1)
        npt.assert_array_equal(b_full[10:], b2)


# ===========================================================================
# weak_form.weak_rows_second_order
# ===========================================================================

def _zero_forcing(t):
    return np.zeros_like(t)


def _unit_forcing(t):
    return np.ones_like(t)


def _trivial_library(states):
    return np.zeros((len(states), 1)), ["dummy"]


class TestWeakRowsSecondOrder:

    def setup_method(self):
        self.time = np.linspace(0.0, 2.0 * np.pi, 8001)
        self.basis = make_test_basis(self.time, num_windows=15)
        u = np.cos(self.time)
        v = -np.sin(self.time)
        self.states = np.column_stack([u, v])

    def test_output_shapes(self):
        A, b, names = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        assert A.shape == (15, 3)   # 15 windows, 1 (u'') + 2 (u, u')
        assert b.shape == (15,)

    def test_first_name_is_udd(self):
        _, _, names = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        assert names[0] == "u''"

    def test_names_append_library_names(self):
        _, _, names = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        assert names == ["u''", "u", "u'"]

    def test_zero_forcing_gives_zero_rhs(self):
        _, b, _ = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        npt.assert_allclose(b, 0.0, atol=1e-12)

    def test_unit_forcing_gives_positive_rhs(self):
        """b_j = ∫1·phi_j dt > 0 since phi_j ≥ 0 and not identically zero."""
        u = np.zeros_like(self.time)
        v = np.zeros_like(self.time)
        states = np.column_stack([u, v])
        _, b, _ = weak_rows_second_order(
            states, self.time, self.basis, _trivial_library, _unit_forcing
        )
        assert np.all(b > 0)

    def test_udd_column_via_ibp(self):
        """First column = -∫v dphi dt (IBP representation of u'')."""
        from scipy.integrate import simpson
        A, _, _ = weak_rows_second_order(
            self.states, self.time, self.basis, _trivial_library, _zero_forcing
        )
        v = self.states[:, 1]
        for j, (phi, dphi) in enumerate(self.basis):
            expected = -simpson(v * dphi, x=self.time)
            assert A[j, 0] == pytest.approx(expected, rel=1e-10)

    def test_library_columns_via_integral(self):
        """Remaining columns = ∫library * phi dt."""
        from scipy.integrate import simpson
        A, _, _ = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        library, _ = _spring_library(self.states)
        for j, (phi, _) in enumerate(self.basis):
            for k in range(library.shape[1]):
                expected = simpson(library[:, k] * phi, x=self.time)
                assert A[j, k + 1] == pytest.approx(expected, rel=1e-10)


# ===========================================================================
# weak_form.build_weak_system_second_order
# ===========================================================================

class TestBuildWeakSystemSecondOrder:

    def setup_method(self):
        self.time = np.linspace(0.0, 2.0 * np.pi, 4001)
        self.basis = make_test_basis(self.time, num_windows=10)
        u = np.cos(self.time)
        v = -np.sin(self.time)
        self.states = np.column_stack([u, v])

    def test_single_trajectory_equals_weak_rows(self):
        A_ref, b_ref, _ = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        A, b, _ = build_weak_system_second_order(
            [self.states], self.time, self.basis, _spring_library, _zero_forcing
        )
        npt.assert_array_equal(A, A_ref)
        npt.assert_array_equal(b, b_ref)

    def test_two_trajectories_double_rows(self):
        u2 = np.sin(self.time)
        v2 = np.cos(self.time)
        s2 = np.column_stack([u2, v2])
        A, b, _ = build_weak_system_second_order(
            [self.states, s2], self.time, self.basis, _spring_library, _zero_forcing
        )
        assert A.shape[0] == 20
        assert b.shape[0] == 20

    def test_row_ordering(self):
        u2 = np.sin(self.time)
        v2 = np.cos(self.time)
        s2 = np.column_stack([u2, v2])
        A_full, b_full, _ = build_weak_system_second_order(
            [self.states, s2], self.time, self.basis, _spring_library, _zero_forcing
        )
        A1, b1, _ = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        npt.assert_array_equal(A_full[:10], A1)
        npt.assert_array_equal(b_full[:10], b1)


# ===========================================================================
# End-to-end SINDy identification
# ===========================================================================

class TestEndToEndSINDy:
    """Recover governing equations from exact, noise-free trajectory data."""

    def test_recover_pure_decay(self):
        """x' = -x  →  coefficient = [-1]."""
        time = np.linspace(0.0, 3.0, 10001)
        states = np.exp(-time)[:, None]
        basis = make_test_basis(time, num_windows=80)
        A, b, _ = build_weak_system([states], time, basis, _identity_library, 0)
        coeffs = CPQR_solver(A, b)
        npt.assert_allclose(coeffs, [-1.0], atol=1e-4)

    def test_recover_affine_ode(self):
        """x' = 2 - x  →  coefficients = [2, -1]."""
        time = np.linspace(0.0, 4.0, 12001)
        x0 = 5.0
        states = (2.0 + (x0 - 2.0) * np.exp(-time))[:, None]
        basis = make_test_basis(time, num_windows=60)
        A, b, _ = build_weak_system([states], time, basis, _affine_library, 0)
        coeffs = CPQR_solver(A, b)
        npt.assert_allclose(coeffs, [2.0, -1.0], atol=1e-3)

    def test_solve_ode_with_true_coefficients(self):
        """Using true coefficients in solve_ode reproduces exact trajectory."""
        time = np.linspace(0.0, 2.0, 1000)
        x0 = 2.0
        traj = solve_ode(_identity_library, np.array([-1.0]), [x0], time)
        npt.assert_allclose(traj[:, 0], x0 * np.exp(-time), rtol=1e-6)

    def test_recover_harmonic_oscillator_homogeneous(self):
        """u'' + u = 0: recover [1, 1] from weak form via SVD null-space method."""
        time = np.linspace(0.0, 4.0 * np.pi, 16001)
        u = np.cos(time)
        v = -np.sin(time)
        states = np.column_stack([u, v])
        basis = make_test_basis(time, num_windows=120)

        def _u_only_library(s):
            return s[:, :1], ["u"]

        # A columns: [u''-IBP, u-integral]; b ≈ 0 (no forcing)
        A, _, _ = weak_rows_second_order(
            states, time, basis, _u_only_library, _zero_forcing
        )
        # True relation: 1·u'' + 1·u = 0 → A @ [1, 1] ≈ 0
        coeffs, _, residual = recover_homogeneous_relation(A, normalization_index=0)
        # coeffs[0] = 1 (normalized), coeffs[1] = omega^2 = 1
        assert coeffs[0] == pytest.approx(1.0, abs=1e-10)
        npt.assert_allclose(coeffs[1], 1.0, atol=1e-3)
        assert residual < 1e-3

    def test_multiple_trajectories_improve_conditioning(self):
        """Using two trajectories gives the same correct answer."""
        time = np.linspace(0.0, 3.0, 6001)
        s1 = np.exp(-time)[:, None]
        s2 = (3.0 * np.exp(-time))[:, None]
        basis = make_test_basis(time, num_windows=40)
        A, b, _ = build_weak_system([s1, s2], time, basis, _identity_library, 0)
        coeffs = CPQR_solver(A, b)
        npt.assert_allclose(coeffs, [-1.0], atol=1e-4)
