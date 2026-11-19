"""Condition-number study for the Duffing weak-form matrix A with respect to [u0; v0]."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wias import build_weak_system_second_order, make_test_basis

ALPHA = 1.0
BETA = 5.0
DELTA = 0.02
GAMMA = 8.0
OMEGA = 0.5


def duffing_forcing(t, gamma=GAMMA, omega=OMEGA):
    return gamma * np.cos(omega * t)


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


def duffing_rhs_true(t, state):
    u, v = state
    du_dt = v
    dv_dt = -DELTA * v - ALPHA * u - BETA * u**3 + GAMMA * np.cos(OMEGA * t)
    return np.array([du_dt, dv_dt], dtype=float)


def generate_trajectory(initial_condition, time):
    solution = solve_ivp(
        duffing_rhs_true,
        (float(time[0]), float(time[-1])),
        np.asarray(initial_condition, dtype=float),
        t_eval=time,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed for {initial_condition}: {solution.message}")
    return solution.y.T


def condition_number_for_state(u0, v0, time, num_basis):
    trajectory = generate_trajectory(np.array([u0, v0], dtype=float), time)
    test_basis = make_test_basis(time, num_basis=num_basis)
    A, _ = build_weak_system_second_order([trajectory], time, test_basis, duffing_library, duffing_forcing)
    return float(np.linalg.cond(A))


def sweep_condition_number(u0_range, v0_range, n_u0, n_v0, time_points, num_basis):
    time_grid = np.linspace(0.0, 30.0, time_points)
    u0_vals = np.linspace(u0_range[0], u0_range[1], n_u0)
    v0_vals = np.linspace(v0_range[0], v0_range[1], n_v0)
    condition_grid = np.empty((n_v0, n_u0), dtype=float)

    total_steps = n_u0 * n_v0
    start_time = time.perf_counter()

    for i, v0 in enumerate(v0_vals):
        for j, u0 in enumerate(u0_vals):
            step_index = i * n_u0 + j
            try:
                condition_grid[i, j] = condition_number_for_state(u0, v0, time_grid, num_basis)
            except Exception:
                condition_grid[i, j] = np.nan

            if step_index == total_steps - 1 or step_index % max(1, total_steps // 20) == 0:
                elapsed = time.perf_counter() - start_time
                progress = (step_index + 1) / total_steps * 100.0
                eta = elapsed * (100.0 / max(progress, 1e-9) - 1.0) if progress > 0 else 0.0
                bar_length = 30
                filled = int(bar_length * progress / 100.0)
                bar = "#" * filled + "-" * (bar_length - filled)
                print(
                    f"[timeline] {bar} {progress:5.1f}% | elapsed {elapsed:6.1f}s | eta {eta:6.1f}s | v0={v0:.3f}, u0={u0:.3f}",
                    flush=True,
                )

    return u0_vals, v0_vals, np.nan_to_num(condition_grid, nan=1e12, posinf=1e12, neginf=1e12)


def plot_condition_grid(u0_vals, v0_vals, condition_grid, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        np.log10(condition_grid),
        extent=[u0_vals[0], u0_vals[-1], v0_vals[0], v0_vals[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("u0")
    ax.set_ylabel("v0")
    ax.set_title("log10(cond(A)) over [u0; v0]")
    cbar = fig.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label("log10(cond(A))")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--u0-range", nargs=2, type=float, default=[-2.0, 2.0])
    parser.add_argument("--v0-range", nargs=2, type=float, default=[-2.0, 2.0])
    parser.add_argument("--n-u0", type=int, default=25)
    parser.add_argument("--n-v0", type=int, default=25)
    parser.add_argument("--time-points", type=int, default=2001)
    parser.add_argument("--num-basis", type=int, default=25)
    parser.add_argument("--output", type=str, default="examples/duffing_condition_number.png")
    args = parser.parse_args()

    u0_vals, v0_vals, cond_grid = sweep_condition_number(
        u0_range=tuple(args.u0_range),
        v0_range=tuple(args.v0_range),
        n_u0=args.n_u0,
        n_v0=args.n_v0,
        time_points=args.time_points,
        num_basis=args.num_basis,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_condition_grid(u0_vals, v0_vals, cond_grid, str(output_path))


if __name__ == "__main__":
    main()
