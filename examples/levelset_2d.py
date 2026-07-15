import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

"""
Plot level set of analytic recovery map $f \mapsto \det(L_f)$ in 2D, where $L_f$ is a 2x2 matrix defined by the function f.
"""


def L_f_matrix(f):
    f0, f1 = f
    L0 = np.array(
        [
            [np.cos((2*f0)**2), np.sin(2*f1+1)],
            [np.sin(2*f1), np.cos(2*f0)],
        ],dtype=float)
    return L0


def psi(f):
    L = L_f_matrix(f)
    return np.linalg.det(L)


x_grid = np.linspace(-2.0, 2.0, 1000)
y_grid = np.linspace(-2, 2, 1000)
F0, F1 = np.meshgrid(x_grid, y_grid)
Psi = np.full_like(F0, np.nan, dtype=float)

for i in range(F0.shape[0]):
    for j in range(F0.shape[1]):
        f = np.array([F0[i, j], F1[i, j]], dtype=float)
        Psi[i, j] = psi(f)

fig, ax = plt.subplots(figsize=(8, 8))

# Zero level contour.
vmax = np.nanmax(np.abs(Psi))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
cf = ax.contourf(F0, F1, Psi, levels=500, cmap='seismic', norm=norm)
cs = ax.contour(F0, F1, Psi, levels=[0], colors='black', linewidths=3.2)

# Horizontal line y = 0.5.
y_target = -0.3
ax.axhline(y_target, color='grey', linestyle='--', linewidth=6)

# Intersections of contour with y = 0.5, extracted from contour segments.
xs = []
for seg in cs.allsegs[0]:
    if len(seg) < 2:
        continue
    for p1, p2 in zip(seg[:-1], seg[1:]):
        x1, y1 = p1
        x2, y2 = p2

        if np.isclose(y1, y_target):
            xs.append(x1)
        if np.isclose(y2, y_target):
            xs.append(x2)

        # Proper crossing between consecutive contour vertices.
        if (y1 - y_target) * (y2 - y_target) < 0:
            t = (y_target - y1) / (y2 - y1)
            x_cross = x1 + t * (x2 - x1)
            xs.append(x_cross)

if xs:
    xs = np.array(xs, dtype=float)
    xs = np.unique(np.round(xs, 6))
    ax.scatter(xs, np.full_like(xs, y_target), s=100, color='black', zorder=5)
    print(f"Intersections with y={y_target}: {len(xs)} points")
else:
    print(f"Intersections with y={y_target}: 0 points")

ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_aspect('equal', adjustable='box')
plt.show()