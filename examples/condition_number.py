import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata

from wias import build_weak_system, make_test_basis

# -------------------------------------------------------------------
# Standard harmonic oscillator on the unit circle
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


# -------------------------------------------------------------------
# Condition number of the recovery matrix vs initial condition
# -------------------------------------------------------------------
def progress_bar(step, total, width=40):
    filled = int((step / total) * width)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {step}/{total}", end="", flush=True)


def condition_number_for_initial(t, x10, x20):
    initial_state = np.array([x10, x20], dtype=float)
    sol = solve_ivp(
        standard_harmonic_on_circle_rhs,
        (float(t[0]), float(t[-1])),
        initial_state,
        t_eval=t,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    trajectory = sol.y.T
    test_basis = make_test_basis(t, num_basis=12)
    A_x1, _ = build_weak_system([trajectory], t, test_basis, standard_harmonic_library, 0)
    A_x2, _ = build_weak_system([trajectory], t, test_basis, standard_harmonic_library, 1)
    A = np.vstack([A_x1, A_x2])
    return float(np.linalg.cond(A))


time = np.linspace(0.0, 80.0, 5001)

# Put more radii near r = 1, where the invariant circle matters most.
radial_anchor = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.2], dtype=float)
radial_fill = np.linspace(0.0, 2.2, 31)
radial_fill = radial_fill[(radial_fill < 0.8) | (radial_fill > 1.2)]
radii = np.unique(np.concatenate([radial_anchor, radial_fill]))
min_points_per_radius = 20
max_points_per_radius = 200

angular_counts = np.maximum(
    min_points_per_radius,
    np.round(
        min_points_per_radius
        + (max_points_per_radius - min_points_per_radius)
        * (0.25 + 0.75 * (radii / radii[-1]) ** 2)
    ).astype(int),
)
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
    cond_values[idx] = condition_number_for_initial(time, float(x10), float(x20))
    progress_bar(idx + 1, initial_pairs.shape[0])
print("\nSweep complete.")

base_cond = np.clip(cond_values[: base_pairs.shape[0]], 1e5, 1e14)
unit_circle_cond = np.clip(cond_values[base_pairs.shape[0] :], 1e5, 1e14)

fig_cond, ax_cond = plt.subplots(figsize=(6, 5))
log_base_cond = np.log10(base_cond)
base_cartesian = np.column_stack([base_pairs[:, 0], base_pairs[:, 1]])
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
)
ax_cond.set_xlim(-1.5, 1.5)
ax_cond.set_ylim(-1.5, 1.5)
ax_cond.set_xlabel(r"$x_1(0)$")
ax_cond.set_ylabel(r"$x_2(0)$")
ax_cond.set_title("Condition number")
colorbar = fig_cond.colorbar(img, ax=ax_cond)
colorbar.set_label(r"$\kappa(A)$")
colorbar.set_ticks(np.log10([1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]))
colorbar.set_ticklabels([
    r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$", r"$10^9$",
    r"$10^{10}$", r"$10^{11}$", r"$10^{12}$", r"$10^{13}$", r"$10^{14}$"
])
fig_cond.tight_layout()
out_cond = "/Users/nboulle/Documents/Git/wias/examples/damped_harmonic_condition_number.png"
fig_cond.savefig(out_cond, dpi=200)
print(f"Saved {out_cond}")
plt.close(fig_cond)
