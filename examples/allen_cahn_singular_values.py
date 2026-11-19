import numpy as np
import scipy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import firedrake as fd

from wias import CPQR_solver, assemble_weak_library_rows


rng = np.random.default_rng(1)


# -------------------------------------------------------------------
# Allen-Cahn library and PDE setup
# -------------------------------------------------------------------
def make_library_exprs(u, v, f):
    library_exprs = {
        "u_xx": -fd.inner(fd.grad(u)[0], fd.grad(v)[0]) * fd.dx,
        "u_xy": -fd.inner(fd.grad(u)[0], fd.grad(v)[1]) * fd.dx,
        "u_yx": -fd.inner(fd.grad(u)[1], fd.grad(v)[0]) * fd.dx,
        "u_yy": -fd.inner(fd.grad(u)[1], fd.grad(v)[1]) * fd.dx,
        "1": fd.Constant(1.0) * v * fd.dx,
        "u": u * v * fd.dx,
        "u^2": u**2 * v * fd.dx,
        "u^3": u**3 * v * fd.dx,
        "u^4": u**4 * v * fd.dx,
        "u^5": u**5 * v * fd.dx,
    }
    rhs = fd.inner(f, v) * fd.dx
    return library_exprs, rhs


def solve_allen_cahn(V, f, true_coefficients):
    u = fd.Function(V, name="u")
    v = fd.TestFunction(V)
    library_exprs, rhs = make_library_exprs(u, v, f)
    a_form = sum(
        fd.Constant(true_coefficients[name]) * expr
        for name, expr in library_exprs.items()
    )
    bc = fd.DirichletBC(V, fd.Constant(0.0), "on_boundary")
    solver_params = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 80,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    fd.solve(a_form - rhs == 0, u, bcs=bc, solver_parameters=solver_params)
    return u


def make_rhs(V, length_scale, amplitude):
    m = V.mesh()
    W = fd.VectorFunctionSpace(m, V.ufl_element())
    X = fd.assemble(fd.interpolate(m.coordinates, W))
    xs = X.dat.data_ro
    n = xs.shape[0]
    dist_sq = scipy.spatial.distance.cdist(xs, xs, "sqeuclidean")
    K = amplitude**2 * np.exp(-dist_sq / (2.0 * length_scale**2))
    K += 1e-8 * np.eye(n)
    L = np.linalg.cholesky(K)
    values = L @ rng.standard_normal(n)
    f = fd.Function(V, name="f")
    f.dat.data[:] = values
    return f


true_coefficients = {
    "u_xx": 0.25,
    "u_xy": 0.0,
    "u_yx": 0.0,
    "u_yy": 0.25,
    "1": 0.0,
    "u": -5.0,
    "u^2": 0.0,
    "u^3": -1.0,
    "u^4": 0.0,
    "u^5": 1.0,
}


# -------------------------------------------------------------------
# Ensemble over 50 random forcing realizations
# -------------------------------------------------------------------
num_realizations = 50
mesh = fd.UnitSquareMesh(100, 100)
V = fd.FunctionSpace(mesh, "CG", 1)

def progress_bar(step, total, width=40):
    filled = int((step / total) * width)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {step}/{total}", end="", flush=True)


singular_values = []
for idx in range(num_realizations):
    f = make_rhs(V, length_scale=0.05, amplitude=10.0)
    u = solve_allen_cahn(V, f, true_coefficients)
    library_exprs, rhs = make_library_exprs(u, fd.TestFunction(V), f)
    bc = fd.DirichletBC(V, fd.Constant(0.0), "on_boundary")
    A, b = assemble_weak_library_rows(library_exprs, rhs, bcs=bc)
    svals = np.linalg.svd(A, compute_uv=False)
    singular_values.append(svals)
    progress_bar(idx + 1, num_realizations)
print("\nEnsemble complete.")

singular_values = np.stack(singular_values)
median_sv = np.median(singular_values, axis=0)
q25_sv = np.quantile(singular_values, 0.25, axis=0)
q75_sv = np.quantile(singular_values, 0.75, axis=0)

# -------------------------------------------------------------------
# Plot median singular values on a log scale with a 25th/75th percentile band
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
idx = np.arange(1, median_sv.size + 1)
ax.plot(
    idx,
    median_sv,
    "o-",
    color="C0",
    linewidth=2.0,
    label="Median singular value",
)
ax.fill_between(
    idx,
    q25_sv,
    q75_sv,
    color="C0",
    alpha=0.25,
    label="25%–75% quartile band",
)
ax.set_xlabel("Singular value index")
ax.set_ylabel("Singular value (log scale)")
ax.set_title("Allen-Cahn matrix singular values across 50 realizations")
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()

out = "/Users/nboulle/Documents/Git/wias/examples/allen_cahn_singular_values.png"
fig.savefig(out, dpi=200)
print(f"Saved {out}")
plt.close(fig)

# Also print the median singular spectrum to the console.
print("Median singular values with 25%/75% quartiles:")
for k, val in enumerate(median_sv, start=1):
    print(f"  sv[{k}] = median {val:.6e}, Q25 {q25_sv[k - 1]:.6e}, Q75 {q75_sv[k - 1]:.6e}")
