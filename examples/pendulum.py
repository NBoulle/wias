import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata

from wias import build_weak_system, build_weak_system_second_order, make_test_basis


# -------------------------------------------------------------------
# Standard harmonic oscillator on the unit circle
# -------------------------------------------------------------------
# We use the nonlinear target system
#   x1' = -x2 + alpha*(r^2 - 1)x1,
#   x2' =  x1 + alpha*(r^2 - 1)x2,
# with r^2 = x1^2 + x2^2 and alpha < 0.
#
# On the invariant circle r^2 = 1, the radial correction vanishes, so the
# dynamics reduces to the standard harmonic oscillator
#   x1' = -x2,
#   x2' =  x1.
#
# Away from the unit circle, the term alpha*(r^2 - 1) acts radially: if
# r > 1, it pushes the trajectory inward; if r < 1, it pushes the trajectory
# outward. This makes the unit circle an attracting invariant manifold, while
# the reduced dynamics on that manifold is still exactly the standard harmonic
# oscillator.
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


def standard_harmonic_forcing(t):
    return 0.0


# -------------------------------------------------------------------
# Representative trajectory on the unit circle
# -------------------------------------------------------------------
time = np.linspace(0.0, 80.0, 5001)
initial = np.array([1.0, 0.0], dtype=float)
solution = solve_ivp(
    standard_harmonic_on_circle_rhs,
    (float(time[0]), float(time[-1])),
    initial,
    t_eval=time,
    method="RK45",
    rtol=1e-10,
    atol=1e-12,
)
traj = solution.y.T

# -------------------------------------------------------------------
# Condition number of the recovery matrix vs initial condition
# -------------------------------------------------------------------
def progress_bar(step, total, width=40):
    filled = int((step / total) * width)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {step}/{total}", end="", flush=True)


def condition_number_for_initial(x10, x20):
    initial_state = np.array([x10, x20], dtype=float)
    sol = solve_ivp(
        standard_harmonic_on_circle_rhs,
        (float(time[0]), float(time[-1])),
        initial_state,
        t_eval=time,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    trajectory = sol.y.T
    test_basis = make_test_basis(time, num_basis=100)
    A_x1, _ = build_weak_system([trajectory], time, test_basis, standard_harmonic_library, 0)
    A_x2, _ = build_weak_system([trajectory], time, test_basis, standard_harmonic_library, 1)
    A = np.vstack([A_x1, A_x2])
    return float(np.linalg.cond(A))


# Put more radii near r = 1, where the invariant circle matters most, while
# still covering the full domain 0 <= r <= 2.2.
radial_anchor = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.2], dtype=float)
radial_fill = np.linspace(0.0, 2.2, 31)
radial_fill = radial_fill[(radial_fill < 0.8) | (radial_fill > 1.2)]
radii = np.unique(np.concatenate([radial_anchor, radial_fill]))
min_points_per_radius = 20
max_points_per_radius = 200

# Increase the angular sampling density with radius, but also keep more angles
# near the center so the interpolation is not dominated by the outer rings.
angular_counts = np.maximum(
    min_points_per_radius,
    np.round(
        min_points_per_radius
        + (max_points_per_radius - min_points_per_radius)
        * (0.25 + 0.75 * (radii / radii[-1]) ** 2)
    ).astype(int),
)
# Add an extra boost near the center for better angular coverage in the inner
# region, where the mesh is otherwise sparse.
angular_counts = np.maximum(angular_counts, np.round(30.0 * (1.0 / (1.0 + radii)) + 20.0).astype(int))
base_pairs = []
for radius, n_angles in zip(radii, angular_counts):
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    base_pairs.append(np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]))
base_pairs = np.vstack(base_pairs)

unit_circle_angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
unit_circle_pairs = np.column_stack([np.cos(unit_circle_angles), np.sin(unit_circle_angles)])
initial_pairs = np.vstack([base_pairs, unit_circle_pairs])
cond_values = np.empty(initial_pairs.shape[0], dtype=float)

for idx, (x10, x20) in enumerate(initial_pairs):
    cond_values[idx] = condition_number_for_initial(float(x10), float(x20))
    progress_bar(idx + 1, initial_pairs.shape[0])
print("\nSweep complete.")
base_cond = np.clip(cond_values[: base_pairs.shape[0]], 1e5, 1e14)
unit_circle_cond = np.clip(cond_values[base_pairs.shape[0] :], 1e5, 1e14)

fig_cond, ax_cond = plt.subplots(figsize=(6, 5))
# Smoothly interpolate the radial samples in Cartesian coordinates.
# The sweep is originally generated in polar form, so convert each point to
# (x, y) before constructing the smooth condition field.
log_base_cond = np.log10(base_cond)
base_cartesian = np.column_stack([
    base_pairs[:, 0],
    base_pairs[:, 1],
])
xs = np.linspace(-1.5, 1.5, 1000)
ys = np.linspace(-1.5, 1.5, 1000)
Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
interp = griddata(base_cartesian, log_base_cond, (Xg, Yg), method="cubic")
interp = np.where(np.isnan(interp), np.log10(1e14), interp)
img = ax_cond.imshow(
    interp,
    extent=[-1.5, 1.5, -1.5, 1.5],
    origin="lower",
    aspect="equal",
    cmap="coolwarm",
    vmin=np.log10(1e5),
    vmax=np.log10(1e14),
    interpolation="bicubic",
)
ax_cond.scatter(
    base_pairs[:, 0],
    base_pairs[:, 1],
    s=8,
    c=np.log10(base_cond),
    cmap="coolwarm",
    vmin=np.log10(1e5),
    vmax=np.log10(1e14),
    alpha=0.9,
    label="sample points",
)
ax_cond.set_xlim(-1.5, 1.5)
ax_cond.set_ylim(-1.5, 1.5)
ax_cond.set_xlabel(r"$x_1(0)$")
ax_cond.set_ylabel(r"$x_2(0)$")
ax_cond.set_title("Condition number")
colorbar = fig_cond.colorbar(img, ax=ax_cond)
colorbar.set_label(r"$\kappa(A)$")
colorbar.set_ticks(np.log10([1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]))
colorbar.set_ticklabels([r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$", r"$10^9$", r"$10^{10}$", r"$10^{11}$", r"$10^{12}$", r"$10^{13}$", r"$10^{14}$"])
fig_cond.tight_layout()
out_cond = "/Users/nboulle/Documents/Git/wias/examples/damped_harmonic_condition_number.png"
fig_cond.savefig(out_cond, dpi=200)
print(f"Saved {out_cond}")
plt.close(fig_cond)

# -------------------------------------------------------------------
# Three phase-plane trajectories: outside, inside, and on the unit circle
# -------------------------------------------------------------------
initials = [
    (np.array([2.0, 0.0]), "outside"),
    (np.array([0.5, 0.0]), "inside"),
    (np.array([1.0, 0.0]), "on"),
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
    if label == "outside":
        axes[0].plot(x, y, lw=2, color="C0", label="outside")
        axes[0].set_title("Outside unit circle")
    elif label == "inside":
        axes[1].plot(x, y, lw=2, color="C1", label="inside")
        axes[1].set_title("Inside unit circle")
    else:
        axes[2].plot(x, y, lw=2, color="C2", label="on")
        axes[2].set_title("On unit circle")

for ax in axes:
    ax.legend(loc="best")

plt.tight_layout()
out1 = "/Users/nboulle/Documents/Git/wias/examples/damped_harmonic_invariant_circle.png"
fig.savefig(out1, dpi=200)
print(f"Saved {out1}")

plt.close(fig)

# -------------------------------------------------------------------
# Actual trajectory vs recovered standard harmonic oscillator from [0, -2]
# -------------------------------------------------------------------
def reduced_harmonic_rhs(t, state):
    x1, x2 = state
    return np.array([-x2, x1], dtype=float)

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
    reduced_harmonic_rhs,
    (float(time[0]), float(time[-1])),
    start,
    t_eval=time,
    method="RK45",
    rtol=1e-10,
    atol=1e-12,
)

# For the nonlinear target system
#   x1' = -x2 + alpha*(r^2 - 1)x1,
#   x2' =  x1 + alpha*(r^2 - 1)x2,
# with alpha = -radial_gain, the coefficients in the library
# [1, x1, x2, x1*r2, x2*r2] are
#   x1': [0, -alpha, -1, alpha, 0]
#   x2': [0, 1, -alpha, 0, alpha]
# and the derivative coefficient itself is fixed to 1 by the first-order form.
fixed_x1_coeffs = np.array([0.0, -alpha, -1.0, alpha, 0.0])
fixed_x2_coeffs = np.array([0.0, 1.0, -alpha, 0.0, alpha])
print("Target coefficients for x1' = ...:", fixed_x1_coeffs)
print("Target coefficients for x2' = ...:", fixed_x2_coeffs)
print("These are the coefficients for the library [1, x1, x2, x1*r2, x2*r2].")

recovery_basis = make_test_basis(time, num_basis=100)
A_x1_fit, b_x1_fit = build_weak_system([actual_sol.y.T], time, recovery_basis, standard_harmonic_library, 0)
A_x2_fit, b_x2_fit = build_weak_system([actual_sol.y.T], time, recovery_basis, standard_harmonic_library, 1)
coeffs_x1, *_ = np.linalg.lstsq(A_x1_fit, b_x1_fit, rcond=None)
coeffs_x2, *_ = np.linalg.lstsq(A_x2_fit, b_x2_fit, rcond=None)
print("Recovered x1' coefficients:", coeffs_x1)
print("Recovered x2' coefficients:", coeffs_x2)

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
    label="recovered (reduced standard oscillator)",
)
ax_comp.set_aspect("equal")
ax_comp.set_xlim(-2.2, 2.2)
ax_comp.set_ylim(-2.2, 2.2)
ax_comp.set_xlabel(r"$x_1$")
ax_comp.set_ylabel(r"$x_2$")
ax_comp.grid(alpha=0.3)
ax_comp.legend(loc="best")
ax_comp.set_title("Trajectory from [0, -2]: actual vs reduced standard oscillator")
plt.tight_layout()
out2 = "/Users/nboulle/Documents/Git/wias/examples/damped_harmonic_from_0_minus_2.png"
fig_comp.savefig(out2, dpi=200)
print(f"Saved {out2}")
plt.close(fig_comp)
