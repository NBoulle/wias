import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from wias import build_weak_system, make_test_basis  # noqa: E402

# -------------------------------------------------------------------
# Standard harmonic oscillator on the unit circle as an invariant manifold
# -------------------------------------------------------------------
radial_gain = 0.01
alpha = -radial_gain


def standard_harmonic_on_circle_rhs(t, state):
    x1, x2 = state
    r2 = x1 * x1 + x2 * x2
    radial = alpha * (r2 - 1.0)
    dx1 = -x2 + radial * x1
    dx2 = x1 + radial * x2
    return np.array([dx1, dx2], dtype=float)


time = np.linspace(0.0, 80.0, 5001)


def standard_harmonic_library(states):
    x1 = states[:, 0]
    x2 = states[:, 1]
    r2 = x1 * x1 + x2 * x2
    columns = np.column_stack([
        np.ones_like(x1),
        x1,
        x2,
        x1 * r2,
        x2 * r2,
    ])
    names = ["1", "x1", "x2", "x1*r2", "x2*r2"]
    return columns, names


def unit_circle_rhs(t, state):
    x1, x2 = state
    return np.array([-x2, x1], dtype=float)


def discovered_reduced_rhs(t, state):
    x1, x2 = state
    return np.array([-x2, x1], dtype=float)


def recovered_rhs_from_coeffs(t, state, coeffs_x1, coeffs_x2):
    x1, x2 = state
    r2 = x1 * x1 + x2 * x2
    library = np.array([1.0, x1, x2, x1 * r2, x2 * r2], dtype=float)
    return np.array([
        float(coeffs_x1 @ library),
        float(coeffs_x2 @ library),
    ], dtype=float)


def coeff_equation_text(coeffs):
    names = ["1", "x_1", "x_2", "x_1 r^2", "x_2 r^2"]
    pieces = []
    for coef, name in zip(coeffs, names):
        if abs(coef) < 1e-12:
            continue
        if abs(coef) < 1.0 - 1e-12 or abs(coef) > 1.0 + 1e-12:
            magnitude = f"{abs(coef):.3g} "
        else:
            magnitude = ""
        if coef < 0:
            sign = "- "
        else:
            sign = "+ " if pieces else ""
        pieces.append(f"{sign}{magnitude}{name}")
    if not pieces:
        return "0"
    return " ".join(pieces)


unit_circle_coeffs_x1 = np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=float)
unit_circle_coeffs_x2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
target_coeffs_x1 = np.array([0.0, -alpha, -1.0, alpha, 0.0], dtype=float)
target_coeffs_x2 = np.array([0.0, 1.0, -alpha, 0.0, alpha], dtype=float)
target_coefficients = np.vstack([target_coeffs_x1, target_coeffs_x2])


# -------------------------------------------------------------------
# Three phase-plane trajectories: outside, inside, and on the unit circle.
# The recovered trajectory is re-simulated from the same initial condition
# using the discovered reduced equation.
# -------------------------------------------------------------------
initials = [
    (np.array([0.5, 0.0]), "inside"),
    (np.array([1.0, 0.0]), "on"),
    (np.array([2.0, 0.0]), "outside"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax in axes:
    unit_circle = plt.Circle((0.0, 0.0), 1.0, fill=False, linestyle="--", color="k", linewidth=1.2)
    ax.add_patch(unit_circle)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(alpha=0.3)

for state0, label in initials:
    sol = solve_ivp(
        standard_harmonic_on_circle_rhs,
        (float(time[0]), float(time[-1])),
        state0,
        t_eval=time,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    x = sol.y[0]
    y = sol.y[1]
    recovered_sol = solve_ivp(
        discovered_reduced_rhs,
        (float(time[0]), float(time[-1])),
        state0,
        t_eval=time,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if label == "outside":
        axes[0].plot(x, y, lw=2, color="C0", label="actual")
        axes[0].plot(recovered_sol.y[0], recovered_sol.y[1], lw=2, color="C3", linestyle="--", label="recovered")
        axes[0].set_title("Outside unit circle")
    elif label == "inside":
        axes[1].plot(x, y, lw=2, color="C1", label="actual")
        axes[1].plot(recovered_sol.y[0], recovered_sol.y[1], lw=2, color="C3", linestyle="--", label="recovered")
        axes[1].set_title("Inside unit circle")
    else:
        axes[2].plot(x, y, lw=2, color="C2", label="actual")
        axes[2].plot(recovered_sol.y[0], recovered_sol.y[1], lw=2, color="C3", linestyle="--", label="recovered")
        axes[2].set_title("On unit circle")

for ax in axes:
    ax.legend(loc="best")

fig.text(
    0.5,
    0.02,
    r"Recovered reduced dynamics: $x_1' = -x_2,\quad x_2' = x_1$",
    ha="center",
    va="center",
    fontsize=12,
)

plt.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
out1 = "/Users/nboulle/Documents/Git/wias/examples/damped_harmonic_invariant_circle.png"
fig.savefig(out1, dpi=200)
print(f"Saved {out1}")
plt.close(fig)

# -------------------------------------------------------------------
# Actual trajectory vs recovered standard harmonic oscillator from [0, -2]
# -------------------------------------------------------------------
start = np.array([0.0, -2.0], dtype=float)
actual_sol = solve_ivp(
    standard_harmonic_on_circle_rhs,
    (float(time[0]), float(time[-1])),
    start,
    t_eval=time,
    method="RK45",
    rtol=1e-10,
    atol=1e-12,
)
recovered_sol = solve_ivp(
    discovered_reduced_rhs,
    (float(time[0]), float(time[-1])),
    start,
    t_eval=time,
    method="RK45",
    rtol=1e-10,
    atol=1e-12,
)

fig_comp, ax_comp = plt.subplots(figsize=(6, 6))
unit_circle = plt.Circle((0.0, 0.0), 1.0, fill=False, linestyle="--", color="k", linewidth=1.2)
ax_comp.add_patch(unit_circle)
ax_comp.plot(actual_sol.y[0], actual_sol.y[1], lw=2, color="C0", label="actual")
ax_comp.plot(
    recovered_sol.y[0],
    recovered_sol.y[1],
    lw=2,
    color="C3",
    linestyle="--",
    label="recovered",
)
ax_comp.set_aspect("equal")
ax_comp.set_xlim(-2.2, 2.2)
ax_comp.set_ylim(-2.2, 2.2)
ax_comp.set_xlabel(r"$x_1$")
ax_comp.set_ylabel(r"$x_2$")
ax_comp.grid(alpha=0.3)
ax_comp.legend(loc="best")
ax_comp.set_title("Trajectory from [0, -2]: actual vs recovered")
plt.tight_layout()
out2 = "/Users/nboulle/Documents/Git/wias/examples/damped_harmonic_from_0_minus_2.png"
fig_comp.savefig(out2, dpi=200)
print(f"Saved {out2}")
plt.close(fig_comp)

# -------------------------------------------------------------------
# Weak-system assembly for the recovered reduced dynamics
# -------------------------------------------------------------------
fig_recovered, axes_recovered = plt.subplots(1, 3, figsize=(15, 5))
for ax in axes_recovered:
    unit_circle = plt.Circle((0.0, 0.0), 1.0, fill=False, linestyle="--", color="k", linewidth=1.2)
    ax.add_patch(unit_circle)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(alpha=0.3)

for state0, label in initials:
    actual_part = solve_ivp(
        standard_harmonic_on_circle_rhs,
        (float(time[0]), float(time[-1])),
        state0,
        t_eval=time,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not actual_part.success:
        raise RuntimeError(actual_part.message)

    recovery_basis = make_test_basis(time, num_basis=100)
    A_x1_fit, b_x1_fit = build_weak_system([actual_part.y.T], time, recovery_basis, standard_harmonic_library, 0)
    A_x2_fit, b_x2_fit = build_weak_system([actual_part.y.T], time, recovery_basis, standard_harmonic_library, 1)
    coeffs_x1_local, *_ = np.linalg.lstsq(A_x1_fit, b_x1_fit, rcond=None)
    coeffs_x2_local, *_ = np.linalg.lstsq(A_x2_fit, b_x2_fit, rcond=None)
    recovered_coefficients = np.vstack([coeffs_x1_local, coeffs_x2_local])
    recovery_error = np.linalg.norm(recovered_coefficients - target_coefficients) / np.linalg.norm(
        target_coefficients
    )
    recovery_matrix = np.vstack([A_x1_fit, A_x2_fit])
    condition_number = np.linalg.cond(recovery_matrix)

    print(f"Trajectory {label}: recovered x1' coefficients = {coeffs_x1_local}")
    print(f"Trajectory {label}: recovered x2' coefficients = {coeffs_x2_local}")
    print(f"Trajectory {label}: relative coefficient error = {recovery_error:.6e}")
    print(f"Trajectory {label}: condition number = {condition_number:.6e}")

    def recovered_rhs(t, state, cx1=coeffs_x1_local, cx2=coeffs_x2_local):
        return recovered_rhs_from_coeffs(t, state, cx1, cx2)

    recovered_part = solve_ivp(
        recovered_rhs,
        (float(time[0]), float(time[-1])),
        state0,
        t_eval=time,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not recovered_part.success:
        raise RuntimeError(recovered_part.message)

    if label == "inside":
        ax = axes_recovered[0]
    elif label == "on":
        ax = axes_recovered[1]
    else:
        ax = axes_recovered[2]

    ax.plot(actual_part.y[0], actual_part.y[1], lw=2, color="blue", label="exact")
    ax.plot(recovered_part.y[0], recovered_part.y[1], lw=2, color="red", linestyle="--", label="recovered")
    title = {"inside": "Inside unit circle", "on": "On unit circle", "outside": "Outside unit circle"}[label]
    ax.set_title(title, pad=18)
    x1_eq = coeff_equation_text(coeffs_x1_local)
    x2_eq = coeff_equation_text(coeffs_x2_local)
    ax.annotate(
        r"$x_1' = " + x1_eq + r"$" + "\n" + r"$x_2' = " + x2_eq + r"$",
        xy=(0.5, 0.0),
        xycoords="axes fraction",
        xytext=(0.0, -28),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=9,
        linespacing=1.3,
        annotation_clip=False,
    )

for ax in axes_recovered:
    ax.legend(loc="best")

fig_recovered.subplots_adjust(bottom=0.18, top=0.88)
out3 = "/Users/nboulle/Documents/Git/wias/examples/damped_harmonic_from_coeffs.png"
fig_recovered.savefig(out3, dpi=200)
print(f"Saved {out3}")
plt.close(fig_recovered)
