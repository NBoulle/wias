"""Duffing recovery error and weak-matrix conditioning versus final time."""

from __future__ import annotations

import argparse
import sys
import time as timer
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("Agg", force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wias import CPQR_solver, build_weak_system_second_order, make_test_basis  # noqa: E402

ALPHA = 1.0
BETA = 5.0
DELTA = 0.02
GAMMA = 8.0
OMEGA = 0.5


def duffing_forcing(time):
    return GAMMA * np.cos(OMEGA * time)


def duffing_library(states):
    u_val = states[:, 0]
    v_val = states[:, 1]
    columns = np.column_stack(
        [
            np.ones_like(u_val),
            u_val,
            v_val,
            u_val**2,
            u_val * v_val,
            v_val**2,
            u_val**3,
            u_val**2 * v_val,
            u_val * v_val**2,
            v_val**3,
        ]
    )
    names = ["1", "u", "v", "u^2", "uv", "v^2", "u^3", "u^2v", "uv^2", "v^3"]
    return columns, names


def duffing_rhs(time, state):
    u_val, v_val = state
    return np.array(
        [
            v_val,
            -DELTA * v_val - ALPHA * u_val - BETA * u_val**3 + GAMMA * np.cos(OMEGA * time),
        ],
        dtype=float,
    )


def solve_trajectory(initial_point, max_final_time):
    solution = solve_ivp(
        duffing_rhs,
        (0.0, max_final_time),
        initial_point,
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"Integration failed for {initial_point}: {solution.message}")
    return solution.sol


def sample_count(final_time, samples_per_unit, num_basis):
    return max(2 * num_basis + 1, int(np.ceil(final_time * samples_per_unit)) + 1)


def recover_coefficients(trajectory, final_time, samples_per_unit, num_basis):
    time_grid = np.linspace(
        0.0,
        final_time,
        sample_count(final_time, samples_per_unit, num_basis),
    )
    states = trajectory(time_grid).T
    test_basis = make_test_basis(time_grid, num_basis=num_basis)
    G, right_hand_side = build_weak_system_second_order(
        [states],
        time_grid,
        test_basis,
        duffing_library,
        duffing_forcing,
    )
    recovered_coefficients = CPQR_solver(G, right_hand_side)
    return recovered_coefficients, float(np.linalg.cond(G))


def recovery_metrics(trajectory, final_time, samples_per_unit, num_basis, true_coefficients):
    recovered_coefficients, condition_number = recover_coefficients(
        trajectory,
        final_time,
        samples_per_unit,
        num_basis,
    )
    relative_error = np.linalg.norm(recovered_coefficients - true_coefficients) / np.linalg.norm(
        true_coefficients
    )
    return float(relative_error), condition_number


def duffing_feature_vector(state):
    u_val, v_val = state
    return np.array(
        [
            1.0,
            u_val,
            v_val,
            u_val**2,
            u_val * v_val,
            v_val**2,
            u_val**3,
            u_val**2 * v_val,
            u_val * v_val**2,
            v_val**3,
        ],
        dtype=float,
    )


def make_recovered_rhs(recovered_coefficients):
    acceleration_coefficient = recovered_coefficients[0]
    if abs(acceleration_coefficient) < 1e-12:
        raise ValueError("Recovered coefficient of u'' is too small to normalize")

    normalized_library_coefficients = -recovered_coefficients[1:] / acceleration_coefficient
    normalized_forcing_coefficient = GAMMA / acceleration_coefficient

    def recovered_rhs(time, state):
        recovered_acceleration = normalized_library_coefficients @ duffing_feature_vector(state)
        recovered_acceleration += normalized_forcing_coefficient * np.cos(OMEGA * time)
        return np.array(
            [
                state[1],
                recovered_acceleration,
            ],
            dtype=float,
        )

    return recovered_rhs


def simulate_system(rhs, initial_point, time_grid):
    solution = solve_ivp(
        rhs,
        (float(time_grid[0]), float(time_grid[-1])),
        initial_point,
        t_eval=time_grid,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    return solution.y.T


def plot_recovered_trajectory(
    initial_point,
    identification_final_time,
    simulation_final_time,
    samples_per_unit,
    num_basis,
    output_path,
):
    exact_dense_trajectory = solve_trajectory(initial_point, simulation_final_time)
    recovered_coefficients, condition_number = recover_coefficients(
        exact_dense_trajectory,
        identification_final_time,
        samples_per_unit,
        num_basis,
    )
    simulation_time = np.linspace(
        0.0,
        simulation_final_time,
        int(np.ceil(simulation_final_time * samples_per_unit)) + 1,
    )
    exact_trajectory = exact_dense_trajectory(simulation_time).T
    recovered_trajectory = simulate_system(
        make_recovered_rhs(recovered_coefficients),
        initial_point,
        simulation_time,
    )

    print(f"Coefficients recovered on [0, {identification_final_time:g}]:")
    print(recovered_coefficients)
    print(f"Condition number at T={identification_final_time:g}: {condition_number:.6e}")

    fig, (displacement_ax, velocity_ax) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    panels = (
        (displacement_ax, 0, "Displacement", "$u(t)$"),
        (velocity_ax, 1, "Velocity", "$u'(t)$"),
    )
    for axis, component, title, ylabel in panels:
        axis.plot(simulation_time, exact_trajectory[:, component], color="C0", linewidth=1.6, label="Exact")
        axis.plot(
            simulation_time,
            recovered_trajectory[:, component],
            color="C3",
            linestyle="--",
            linewidth=1.2,
            label="Recovered",
        )
        axis.axvline(
            identification_final_time,
            color="0.3",
            linestyle=":",
            linewidth=1.2,
            label="End of identification window",
        )
        axis.set_title(title)
        axis.set_xlabel("Time $t$")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()

    displacement_ax.set_xlim(0.0, simulation_final_time)
    fig.suptitle(
        f"Duffing trajectory recovered on $[0,{identification_final_time:g}]$ "
        f"and simulated to $T={simulation_final_time:g}$"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_ensemble(
    final_times,
    initial_points,
    samples_per_unit,
    num_basis,
):
    true_coefficients = np.array(
        [1.0, 0.0, ALPHA, DELTA, 0.0, 0.0, 0.0, BETA, 0.0, 0.0, 0.0],
        dtype=float,
    )
    errors = np.empty((len(initial_points), len(final_times)), dtype=float)
    condition_numbers = np.empty_like(errors)
    total_steps = len(initial_points) * len(final_times)
    completed_steps = 0
    start = timer.perf_counter()

    for point_index, initial_point in enumerate(initial_points):
        trajectory = solve_trajectory(initial_point, float(final_times[-1]))
        for time_index, final_time in enumerate(final_times):
            errors[point_index, time_index], condition_numbers[point_index, time_index] = recovery_metrics(
                trajectory,
                float(final_time),
                samples_per_unit,
                num_basis,
                true_coefficients,
            )
            completed_steps += 1
            if completed_steps == total_steps or completed_steps % max(1, total_steps // 20) == 0:
                elapsed = timer.perf_counter() - start
                print(
                    f"[{completed_steps:4d}/{total_steps}] "
                    f"initial point {point_index + 1}/{len(initial_points)}, "
                    f"T={final_time:6.2f}, elapsed={elapsed:7.1f}s",
                    flush=True,
                )

    return errors, condition_numbers


def quartiles(values):
    return (
        np.quantile(values, 0.25, axis=0),
        np.median(values, axis=0),
        np.quantile(values, 0.75, axis=0),
    )


def plot_results(final_times, errors, condition_numbers, output_path):
    error_q25, error_median, error_q75 = quartiles(errors)
    cond_q25, cond_median, cond_q75 = quartiles(condition_numbers)

    fig, (error_ax, cond_ax) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    error_ax.plot(final_times, error_median, color="C0", linewidth=2.0, label="Median")
    error_ax.fill_between(
        final_times,
        error_q25,
        error_q75,
        color="C0",
        alpha=0.25,
        label="25%–75% quartile band",
    )
    error_ax.set_title("Coefficient recovery error")
    error_ax.set_xlabel("Final time $T$")
    error_ax.set_ylabel(r"Relative error $\|\widehat{c}-c\|_2/\|c\|_2$")
    error_ax.set_yscale("log")
    error_ax.grid(True, which="both", alpha=0.3)
    error_ax.legend()

    cond_ax.plot(final_times, cond_median, color="C1", linewidth=2.0, label="Median")
    cond_ax.fill_between(
        final_times,
        cond_q25,
        cond_q75,
        color="C1",
        alpha=0.25,
        label="25%–75% quartile band",
    )
    cond_ax.set_title("Condition number of $G$")
    cond_ax.set_xlabel("Final time $T$")
    cond_ax.set_ylabel(r"$\kappa_2(G)$")
    cond_ax.set_yscale("log")
    cond_ax.grid(True, which="both", alpha=0.3)
    cond_ax.legend()

    error_ax.set_xlim(0.0, float(final_times[-1]))
    fig.suptitle(f"Duffing identification over {errors.shape[0]} random initial points")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-final-time", type=float, default=100.0)
    parser.add_argument("--num-final-times", type=int, default=50)
    parser.add_argument("--num-initial-points", type=int, default=10)
    parser.add_argument("--initial-low", type=float, default=-2.0)
    parser.add_argument("--initial-high", type=float, default=2.0)
    parser.add_argument("--samples-per-unit", type=int, default=100)
    parser.add_argument("--num-basis", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identification-final-time", type=float, default=5.0)
    parser.add_argument("--simulation-final-time", type=float, default=100.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/duffing_final_time_quartiles.png"),
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=Path("examples/duffing_recovered_trajectory.png"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_final_time <= 0.0:
        raise ValueError("--max-final-time must be positive")
    if args.identification_final_time <= 0.0:
        raise ValueError("--identification-final-time must be positive")
    if args.simulation_final_time < args.identification_final_time:
        raise ValueError("--simulation-final-time must not precede the identification final time")
    if args.num_final_times < 1 or args.num_initial_points < 1:
        raise ValueError("The numbers of final times and initial points must be positive")
    if args.num_basis < 11:
        raise ValueError("--num-basis must be at least 11 for the Duffing library")
    if args.samples_per_unit < 2:
        raise ValueError("--samples-per-unit must be at least 2")
    if args.initial_low >= args.initial_high:
        raise ValueError("--initial-low must be smaller than --initial-high")

    final_times = np.linspace(0.0, args.max_final_time, args.num_final_times + 1)[1:]
    random_generator = np.random.default_rng(args.seed)
    initial_points = random_generator.uniform(
        args.initial_low,
        args.initial_high,
        size=(args.num_initial_points, 2),
    )
    print("Random initial points [u(0), u'(0)]:")
    print(initial_points)

    errors, condition_numbers = run_ensemble(
        final_times,
        initial_points,
        args.samples_per_unit,
        args.num_basis,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_results(final_times, errors, condition_numbers, args.output)
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    plot_recovered_trajectory(
        initial_points[0],
        args.identification_final_time,
        args.simulation_final_time,
        args.samples_per_unit,
        args.num_basis,
        args.trajectory_output,
    )


if __name__ == "__main__":
    main()
