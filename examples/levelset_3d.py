import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


"""
Plot level set of analytic recovery map $f \mapsto \det(L_f)$ in 3D, where $L_f$ is a 3x3 matrix defined by the function f.
"""


def L_f_matrix(f):
    f0, f1, f2 = f
    L = np.array(
        [
            [(1-(0.2*f2**2)), 2*f1, 0],
            [f1, np.sin((3*f0**3))*np.cos(0.5*f2)**2*(3-f2), 0],
            [0, 0, 1]
        ], dtype=float)
    return L


def psi(f):
    L = L_f_matrix(f)
    return np.linalg.det(L)


def psi_on_tensor_grid(X, Y, Z):
    """Evaluate psi(x, y, z) on tensor-product arrays X, Y, Z."""
    a = 1.0 - 0.2 * Z**2
    b = np.sin(3.0 * X**3) * np.cos(0.5 * Z) ** 2 * (3.0 - Z)
    return a * b - 2.0 * Y**2


def contour_segments_from_matrix(F0, F1, matrix):
    """Extract zero-contour segments of a 2D scalar field matrix=0."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(F0, F1, matrix, levels=[0], colors='black', linewidths=3.2)
    segs = cs.allsegs[0]
    plt.close(fig_tmp)
    return segs


def _resample_segment(seg, n_samples=220):
    """Resample a polyline segment to n_samples points by arc length."""
    seg = np.asarray(seg, dtype=float)
    if len(seg) < 2:
        return np.repeat(seg[:1], n_samples, axis=0)

    d = np.sqrt(np.sum(np.diff(seg, axis=0) ** 2, axis=1))
    s = np.concatenate(([0.0], np.cumsum(d)))
    if s[-1] <= 1e-14:
        return np.repeat(seg[:1], n_samples, axis=0)

    s_new = np.linspace(0.0, s[-1], n_samples)
    x_new = np.interp(s_new, s, seg[:, 0])
    y_new = np.interp(s_new, s, seg[:, 1])
    return np.column_stack([x_new, y_new])


def _sort_segments_by_centroid(segs):
    """Sort segments so adjacent z-slices can be paired consistently."""
    segs = [np.asarray(seg, dtype=float) for seg in segs if len(seg) >= 2]
    return sorted(segs, key=lambda seg: (np.mean(seg[:, 0]), np.mean(seg[:, 1])))


# 3D example: contour slices in (f0, f1) for varying f2=z.
print("Computing contour slices for f -> det(L_f)")

r = 1001

x_grid = np.linspace(-2.0, 2.0, r)
y_grid = np.linspace(-2, 2, r)
z_min, z_max = -2.0, 2.0
nz = 201
z_vals = np.linspace(z_min, z_max, nz)

F0, F1 = np.meshgrid(x_grid, y_grid)
X3, Y3, Z3 = np.meshgrid(x_grid, y_grid, z_vals, indexing='xy')
Psi3 = psi_on_tensor_grid(X3, Y3, Z3)

z_plane = 0.0
z0_idx = int(np.argmin(np.abs(z_vals - z_plane)))
Psi = Psi3[:, :, z0_idx]
cs_plane = contour_segments_from_matrix(F0, F1, Psi)

print(f"Grid points evaluated: {Psi3.size}")
print(f"min psi(z=0) = {np.nanmin(Psi):.3e}, max psi(z=0) = {np.nanmax(Psi):.3e}")

vmax = np.nanmax(np.abs(Psi))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
xy_ticks = [-2, -1, 0, 1, 2]

# Second plot: z-invariant contour field (x, y, z) -> contour(x, y).
fig3d = plt.figure(figsize=(9, 7))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.computed_zorder = False

# Build contours on all z-slices, then connect adjacent slices with ribbons.
slice_segs = []
for k in range(len(z_vals)):
    segs = contour_segments_from_matrix(F0, F1, Psi3[:, :, k])
    slice_segs.append(_sort_segments_by_centroid(segs))

def draw_branch_surfaces(slice_indices, zorder):
    """Draw one continuous surface per contour branch across many z-slices."""
    # Keep only slices that actually have contour branches.
    valid_indices = [k for k in slice_indices if len(slice_segs[k]) > 0]
    if len(valid_indices) < 2:
        return

    n_branch = min(len(slice_segs[k]) for k in valid_indices)
    if n_branch == 0:
        return

    n_samples = 101
    for i in range(n_branch):
        # Fill area between consecutive z-slice contours.
        for j in range(len(valid_indices) - 1):
            k0 = valid_indices[j]
            k1 = valid_indices[j + 1]

            seg0 = _resample_segment(slice_segs[k0][i], n_samples=n_samples)
            seg1 = _resample_segment(slice_segs[k1][i], n_samples=n_samples)

            poly = ax3d.fill_between(
                seg0[:, 0],
                seg0[:, 1],
                np.full(n_samples, z_vals[k0], dtype=float),
                seg1[:, 0],
                seg1[:, 1],
                np.full(n_samples, z_vals[k1], dtype=float),
                facecolors=(0.0, 0.0, 0.0, 0.18),
                shade=False,
                edgecolor='none',
                linewidth=0,
            )
            poly.set_zorder(zorder)


# Plot negative-z surfaces first.
neg_indices = [k for k, z in enumerate(z_vals) if z <= 0]
draw_branch_surfaces(neg_indices, zorder=0)

# Add the first figure as a z=0 plane in the 3D scene.
cf3d = ax3d.contourf(
    F0,
    F1,
    Psi,
    levels=500,
    cmap='seismic',
    norm=norm,
    zdir='z',
    offset=z_plane,
    alpha=1,
    zorder=1,
)

# Plot positive-z surfaces last.
pos_indices = [k for k, z in enumerate(z_vals) if z >= 0]
draw_branch_surfaces(pos_indices, zorder=3)

# Draw a fully opaque zero level set slightly above the plane for visibility.
# Using explicit 3D lines avoids transparency/depth-blending artifacts from mplot3d contour.
z_halo = z_plane + 0
for seg in cs_plane:
    if len(seg) < 2:
        continue
    ax3d.plot(
        seg[:, 0],
        seg[:, 1],
        np.full(seg.shape[0], z_halo),
        color='black',
        linewidth=2,
        alpha=1.0,
        antialiased=False,
        solid_capstyle='round',
        zorder=2,
    )


ax3d.set_xlabel(r'$x$')
ax3d.set_ylabel(r'$y$')
ax3d.set_zlabel(r'$z$')
ax3d.set_xlim([-2, 2])
ax3d.set_ylim([-2, 2])
ax3d.set_zlim([z_min, z_max])
ax3d.view_init(elev=25, azim=-110)

plt.tight_layout()
# Additional exports without axes for TikZ overlay in LaTeX.
fig3d.patch.set_alpha(0)
ax3d.set_facecolor((1, 1, 1, 0))
plt.show()
