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
        self.T = self.time[-1] - self.time[0]
        self.t_min = self.time[0]

    def test_correct_number_of_basis_functions(self):
        assert len(self.basis) == self.n

    def test_arrays_have_correct_shape(self):
        for phi, dphi in self.basis:
            assert phi.shape == self.time.shape
            assert dphi.shape == self.time.shape

    def test_phi_vanishes_at_endpoints(self):
        """phi_k = sin(k*pi*(t - t_min)/T) vanishes at both endpoints."""
        for phi, _ in self.basis:
            assert phi[0] == pytest.approx(0.0, abs=1e-14)
            assert phi[-1] == pytest.approx(0.0, abs=1e-14)

    def test_phi_formula(self):
        """phi_k = sin(k*pi*(t - t_min) / T)."""
        for k, (phi, _) in enumerate(self.basis, start=1):
            omega = k * np.pi / self.T
            expected = np.sin(omega * (self.time - self.t_min))
            npt.assert_allclose(phi, expected, atol=1e-14)

    def test_dphi_formula(self):
        """dphi_k = k*pi/T * cos(k*pi*(t - t_min) / T)."""
        for k, (_, dphi) in enumerate(self.basis, start=1):
            omega = k * np.pi / self.T
            expected = omega * np.cos(omega * (self.time - self.t_min))
            npt.assert_allclose(dphi, expected, atol=1e-14)

    def test_dphi_is_derivative_of_phi(self):
        """dphi ≈ d/dt phi via finite differences (interior points)."""
        time_fine = np.linspace(0.0, 1.0, 100001)
        basis_fine = make_test_basis(time_fine, 3)
        for phi, dphi in basis_fine:
            fd = np.gradient(phi, time_fine)
            npt.assert_allclose(dphi[1:-1], fd[1:-1], atol=1e-5)

    def test_orthogonality_different_frequencies(self):
        """Consecutive sinusoidal modes are L2-orthogonal on the interval."""
        from scipy.integrate import simpson as simp
        phi1, _ = self.basis[0]
        phi2, _ = self.basis[1]
        inner = simp(phi1 * phi2, x=self.time)
        assert inner == pytest.approx(0.0, abs=1e-10)

    def test_varying_num_basis(self):
        """make_test_basis returns the requested number of functions."""
        for n in [1, 5, 20]:
            basis = make_test_basis(self.time, n)
            assert len(basis) == n


# ===========================================================================
# weak_form.weak_library_rows — integration-by-parts correctness
# ===========================================================================

class TestWeakLibraryRows:

    def setup_method(self):
        self.time = np.linspace(0.0, 1.0, 8001)
        self.basis = make_test_basis(self.time, 20)

    def test_output_shapes(self):
        states = np.sin(self.time)[:, None]
        A, b = weak_library_rows(states, self.time, self.basis, _affine_library, 0)
        assert A.shape == (20, 2)
        assert b.shape == (20,)

    def test_ibp_linear_trajectory(self):
        """x(t) = t → dx/dt = 1 = 1·1 + 0·x.  A @ [1, 0] ≈ b via IBP."""
        states = self.time[:, None]
        A, b = weak_library_rows(states, self.time, self.basis, _affine_library, 0)
        npt.assert_allclose(A @ np.array([1.0, 0.0]), b, atol=1e-5)

    def test_ibp_exponential_trajectory(self):
        """x(t) = e^t → dx/dt = x = 1·x.  A @ [1] ≈ b via IBP."""
        x = np.exp(self.time)
        states = x[:, None]
        A, b = weak_library_rows(states, self.time, self.basis, _identity_library, 0)
        npt.assert_allclose(A @ np.array([1.0]), b, atol=1e-4)

    def test_ibp_constant_trajectory_gives_zero_rhs(self):
        """x(t) = 1 → dx/dt = 0 → b_j = -∫dphi_j dt = -[phi_j]_boundary = 0.

        The sinusoidal test functions vanish at both endpoints, so the boundary
        term is exactly zero. Simpson's rule introduces a small numerical
        quadrature error on the order of 1e-6.
        """
        states = np.ones((len(self.time), 1))
        _, b = weak_library_rows(states, self.time, self.basis, _constant_library, 0)
        npt.assert_allclose(b, 0.0, atol=2e-5)

    def test_correct_component_index(self):
        """component_index selects the right column of states."""
        # Two-state system: component_index=1 should use states[:,1]
        x = self.time[:, None]
        y = np.exp(self.time)[:, None]
        states = np.hstack([x, y])
        _, b_comp1 = weak_library_rows(states, self.time, self.basis, _identity_library, 1)
        # For y(t)=e^t with library [y]: b_j = -∫e^t dphi_j dt ≈ ∫e^t phi_j dt (IBP)
        states_exp = y
        A_ref, b_ref = weak_library_rows(states_exp, self.time, self.basis, _identity_library, 0)
        npt.assert_allclose(b_comp1, b_ref, atol=1e-10)

    def test_names_from_library(self):
        states = np.ones((len(self.time), 1))
        _, names = _affine_library(states)
        assert names == ["1", "x"]


# ===========================================================================
# weak_form.build_weak_system
# ===========================================================================

class TestBuildWeakSystem:

    def setup_method(self):
        self.time = np.linspace(0.0, 1.0, 4001)
        self.basis = make_test_basis(self.time, 10)

    def test_single_trajectory_equals_weak_library_rows(self):
        states = np.exp(self.time)[:, None]
        A_ref, b_ref = weak_library_rows(states, self.time, self.basis, _affine_library, 0)
        A, b = build_weak_system([states], self.time, self.basis, _affine_library, 0)
        npt.assert_array_equal(A, A_ref)
        npt.assert_array_equal(b, b_ref)

    def test_two_trajectories_stack_rows(self):
        s1 = np.exp(self.time)[:, None]
        s2 = np.sin(self.time)[:, None]
        A, b = build_weak_system([s1, s2], self.time, self.basis, _affine_library, 0)
        assert A.shape == (20, 2)    # 10 windows × 2 trajectories
        assert b.shape == (20,)

    def test_names_consistent_across_trajectories(self):
        s1 = np.exp(self.time)[:, None]
        _, names = _affine_library(s1)
        assert names == ["1", "x"]

    def test_row_order_first_trajectory_then_second(self):
        """First 10 rows come from trajectory 1; next 10 from trajectory 2."""
        s1 = np.exp(self.time)[:, None]
        s2 = np.sin(self.time)[:, None]
        A_full, b_full = build_weak_system([s1, s2], self.time, self.basis, _affine_library, 0)
        A1, b1 = weak_library_rows(s1, self.time, self.basis, _affine_library, 0)
        A2, b2 = weak_library_rows(s2, self.time, self.basis, _affine_library, 0)
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
        self.basis = make_test_basis(self.time, 15)
        u = np.cos(self.time)
        v = -np.sin(self.time)
        self.states = np.column_stack([u, v])

    def test_output_shapes(self):
        A, b = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        assert A.shape == (15, 3)   # 15 basis functions, 1 (u'') + 2 (u, u')
        assert b.shape == (15,)

    def test_column_count_is_one_plus_library_size(self):
        """A has 1 + len(library) columns (u'' column + library columns)."""
        A, _ = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        _, lib_names = _spring_library(self.states)
        assert A.shape[1] == 1 + len(lib_names)

    def test_library_names_from_function(self):
        """Library function returns expected term names."""
        _, names = _spring_library(self.states)
        assert names == ["u", "u'"]

    def test_zero_forcing_gives_zero_rhs(self):
        _, b = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        npt.assert_allclose(b, 0.0, atol=1e-12)

    def test_forcing_rhs_matches_integral(self):
        """b_j = \u222b f(t) * phi_j dt for each basis function."""
        from scipy.integrate import simpson as simp
        u = np.zeros_like(self.time)
        v = np.zeros_like(self.time)
        states = np.column_stack([u, v])
        _, b = weak_rows_second_order(
            states, self.time, self.basis, _trivial_library, _unit_forcing
        )
        for j, (phi, _) in enumerate(self.basis):
            expected = simp(phi, x=self.time)
            assert b[j] == pytest.approx(expected, rel=1e-10, abs=1e-10)

    def test_udd_column_via_ibp(self):
        """First column = -∫v dphi dt (IBP representation of u'')."""
        from scipy.integrate import simpson
        A, _ = weak_rows_second_order(
            self.states, self.time, self.basis, _trivial_library, _zero_forcing
        )
        v = self.states[:, 1]
        for j, (phi, dphi) in enumerate(self.basis):
            expected = -simpson(v * dphi, x=self.time)
            assert A[j, 0] == pytest.approx(expected, rel=1e-10)

    def test_library_columns_via_integral(self):
        """Remaining columns = ∫library * phi dt."""
        from scipy.integrate import simpson
        A, _ = weak_rows_second_order(
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
        self.basis = make_test_basis(self.time, 10)
        u = np.cos(self.time)
        v = -np.sin(self.time)
        self.states = np.column_stack([u, v])

    def test_single_trajectory_equals_weak_rows(self):
        A_ref, b_ref = weak_rows_second_order(
            self.states, self.time, self.basis, _spring_library, _zero_forcing
        )
        A, b = build_weak_system_second_order(
            [self.states], self.time, self.basis, _spring_library, _zero_forcing
        )
        npt.assert_array_equal(A, A_ref)
        npt.assert_array_equal(b, b_ref)

    def test_two_trajectories_double_rows(self):
        u2 = np.sin(self.time)
        v2 = np.cos(self.time)
        s2 = np.column_stack([u2, v2])
        A, b = build_weak_system_second_order(
            [self.states, s2], self.time, self.basis, _spring_library, _zero_forcing
        )
        assert A.shape[0] == 20
        assert b.shape[0] == 20

    def test_row_ordering(self):
        u2 = np.sin(self.time)
        v2 = np.cos(self.time)
        s2 = np.column_stack([u2, v2])
        A_full, b_full = build_weak_system_second_order(
            [self.states, s2], self.time, self.basis, _spring_library, _zero_forcing
        )
        A1, b1 = weak_rows_second_order(
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
        basis = make_test_basis(time, 80)
        A, b = build_weak_system([states], time, basis, _identity_library, 0)
        coeffs = CPQR_solver(A, b)
        npt.assert_allclose(coeffs, [-1.0], atol=1e-4)

    def test_recover_affine_ode(self):
        """x' = 2 - x  →  coefficients = [2, -1]."""
        time = np.linspace(0.0, 4.0, 12001)
        x0 = 5.0
        states = (2.0 + (x0 - 2.0) * np.exp(-time))[:, None]
        basis = make_test_basis(time, 60)
        A, b = build_weak_system([states], time, basis, _affine_library, 0)
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
        basis = make_test_basis(time, 120)

        def _u_only_library(s):
            return s[:, :1], ["u"]

        # A columns: [u''-IBP, u-integral]; b ≈ 0 (no forcing)
        A, _ = weak_rows_second_order(
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
        basis = make_test_basis(time, 40)
        A, b = build_weak_system([s1, s2], time, basis, _identity_library, 0)
        coeffs = CPQR_solver(A, b)
        npt.assert_allclose(coeffs, [-1.0], atol=1e-4)
