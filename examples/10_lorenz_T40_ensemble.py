"""Lorenz empirical coverage at T=40, matching the referee-response ensemble.

Recovery uses the WIAS test basis, weak-system assembly, and CPQR solver.
Outputs are written to examples/lorenz_T40_ensemble/ by default.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lorenz-t40-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wias import CPQR_solver, build_weak_system, make_test_basis  # noqa: E402

FINAL_TIME = 40  # Fixed for every trajectory and quadrature resolution.
TRUE = np.array([[0., -10., 10., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 28., -1., 0., 0., 0., -1., 0., 0., 0.],
                 [0., 0., 0., -8/3, 0., 1., 0., 0., 0., 0.]])
FEATURE_NAMES = ["1", "x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]


def rhs(t, state):
    x, y, z = state
    return [10*(y-x), x*(28-z)-y, x*y-(8/3)*z]


def library(states):
    x, y, z = states.T
    return np.column_stack([np.ones_like(x), x, y, z, x*x, x*y, x*z, y*y, y*z, z*z]), FEATURE_NAMES


def recover_coefficients(trajectory, samples_per_unit, num_basis=50):
    """Recover each Lorenz component on the fixed interval [0, 40]."""
    time_grid = np.linspace(
        0.0,
        FINAL_TIME,
        max(2 * num_basis + 1, FINAL_TIME * samples_per_unit + 1),
    )
    states = trajectory(time_grid).T
    test_basis = make_test_basis(time_grid, num_basis=num_basis)
    coeff = np.empty_like(TRUE)
    b = np.empty((len(test_basis), 3))
    g = None
    for component in range(3):
        matrix, right_hand_side = build_weak_system(
            [states], time_grid, test_basis, library, component
        )
        with contextlib.redirect_stdout(io.StringIO()):
            coeff[component] = CPQR_solver(matrix, right_hand_side)
        b[:, component] = right_hand_side
        if g is None:
            g = matrix
    singular_values = np.linalg.svd(g, compute_uv=False)
    diagonal = np.abs(np.diag(qr(g, mode="economic", pivoting=True)[1]))
    rank = int(np.count_nonzero(diagonal > 1e-10*diagonal[0]))
    error = np.linalg.norm(coeff-TRUE)/np.linalg.norm(TRUE)
    condition = singular_values[0]/singular_values[-1] if singular_values[-1] else np.inf
    return dict(g=g, b=b, singular_values=singular_values, rank=rank, coefficients=coeff,
                condition=float(condition), error=float(error))


def upward(value, significant=4):
    """Round upward so the printed value remains a bound on the samples."""
    if value == 0 or not np.isfinite(value):
        return float(value)
    step = 10.**(math.floor(math.log10(value))-significant+1)
    return float(math.ceil(value/step)*step)


def write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def figures(conditions, errors, output):
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    for ax, values, title, ylabel, color in zip(
            axes, [conditions, errors], ["Recovery-matrix condition number", "Coefficient recovery error"],
            [r"Condition-number bound $K$", r"Relative-error bound $\varepsilon$"], ["#c66a18", "#2468a0"]):
        ordered = np.sort(values)
        x = np.r_[0., np.arange(1, len(values)+1)*100/len(values)]
        y = np.r_[ordered[0], ordered]
        ax.step(x, y, where="pre", color=color, linewidth=2.)
        ax.set(yscale="linear", xlim=(0, 100), ylim=(ordered[0]/1.1, ordered[-1]*1.1), ylabel=ylabel,
               xlabel="Percentage of sampled inputs (%)", title=title)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True, useOffset=False)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.grid(alpha=.2)
    fig.subplots_adjust(left=.075, right=.985, bottom=.24, top=.79, wspace=.28)
    fig.savefig(output/"lorenz_T40_condition_and_error_bounds.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"lorenz_T40_ensemble")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [PROJECT_ROOT/"wias/weak_form.py", PROJECT_ROOT/"wias/linear_algebra.py"]
    originals = sources
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    seed = 42
    inputs = np.random.default_rng(seed).uniform(-10., 10., (100, 3))
    rates = [100, 200, 1000]
    conditions, errors, ranks = np.empty((100, 3)), np.empty((100, 3)), np.empty((100, 3), dtype=int)
    coefficients, singular_values = np.empty((100, 3, 3, 10)), np.empty((100, 3, 10))
    elapsed = time.perf_counter()
    for index, initial in enumerate(inputs):
        solution = solve_ivp(rhs, (0., FINAL_TIME), initial, method="DOP853", dense_output=True, rtol=1e-10, atol=1e-12)
        if not solution.success:
            raise RuntimeError(f"Trajectory {index} failed: {solution.message}")
        for j, rate in enumerate(rates):
            result = recover_coefficients(solution.sol, rate)
            conditions[index, j], errors[index, j], ranks[index, j] = result["condition"], result["error"], result["rank"]
            coefficients[index, j], singular_values[index, j] = result["coefficients"], result["singular_values"]
        if (index+1) % 10 == 0:
            print(f"{index+1}/100 trajectories, three quadrature resolutions: {time.perf_counter()-elapsed:.1f}s", flush=True)
    assert np.isfinite(conditions).all() and np.isfinite(errors).all()
    np.savez_compressed(args.output/"lorenz_T40_samples.npz", inputs=inputs, conditions=conditions, errors=errors,
                        ranks=ranks, coefficients=coefficients, singular_values=singular_values,
                        true_coefficients=TRUE, samples_per_unit=np.array(rates))
    write_csv(args.output/"samples.csv", ["sample", "x0", "y0", "z0", "samples_per_unit", "condition", "relative_coefficient_error", "numerical_rank"],
              [[i+1, *inputs[i], rate, conditions[i,j], errors[i,j], ranks[i,j]] for i in range(100) for j,rate in enumerate(rates)])
    coverage = []
    for percent in [50, 90, 95, 99, 100]:
        k = math.ceil(percent)-1
        condition = upward(np.sort(conditions[:, 0])[k])
        error = upward(np.sort(errors[:, 0])[k])
        coverage.append(dict(percent=percent, condition_bound=condition, error_bound=error,
                             condition_count=int(np.count_nonzero(conditions[:,0]<=condition)),
                             error_count=int(np.count_nonzero(errors[:,0]<=error)),
                             joint_count=int(np.count_nonzero((conditions[:,0]<=condition)&(errors[:,0]<=error)))))
    write_csv(args.output/"coverage_bounds.csv", list(coverage[0]), [list(row.values()) for row in coverage])
    figures(conditions[:,0], errors[:,0], args.output)
    refinement = [{"samples_per_unit": rate, "median_condition": float(np.median(conditions[:,j])),
                   "maximum_condition": float(np.max(conditions[:,j])), "median_error": float(np.median(errors[:,j])),
                   "maximum_error": float(np.max(errors[:,j])),
                   "max_condition_relative_change_from_primary": float(np.max(abs(conditions[:,j]/conditions[:,0]-1))),
                   "minimum_numerical_rank": int(np.min(ranks[:,j]))} for j,rate in enumerate(rates)]
    for name, digest in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, f"External file changed during run: {name}"
    summary = dict(T=float(FINAL_TIME), samples=100, seed=seed, law="Independent uniform on [-10,10]^3", dictionary=FEATURE_NAMES,
                   tests=f"sin(k*pi*t/{FINAL_TIME}), k=1,...,50; no column rescaling", primary_dt=.01,
                   solver="DOP853", rtol=1e-10, atol=1e-12, rank_tolerance=1e-10,
                   error="Frobenius norm(C_hat-C)/Frobenius norm(C)", coverage=coverage,
                   refinement=refinement, recovery="WIAS build_weak_system and CPQR_solver per component",
                   original_files_sha256=hashes,
                   wias_sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                   numpy=np.__version__, scipy=scipy.__version__, elapsed_seconds=time.perf_counter()-elapsed,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"coverage":coverage,"refinement":refinement}, indent=2), flush=True)


if __name__ == "__main__":
    main()
