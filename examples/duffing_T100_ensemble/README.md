# Duffing empirical coverage at T=100

Run `python3 examples/11_duffing_T100_ensemble.py` from the repository root.

The experiment uses 100 independent initial conditions uniform on [-2,2]^2
(seed 42), the forced Duffing parameters from `08_duffing_final_time.py`,
and a fixed observation interval [0,100]. Recovery directly calls WIAS
`make_test_basis`, `build_weak_system_second_order`, and `CPQR_solver`.
The 11-column matrix includes acceleration and the ten cubic polynomial
terms. The coefficient vector is [1,0,1,0.02,0,0,0,5,0,0,0].

DOP853 uses rtol=1e-10 and atol=1e-12. There are 50 sine test functions,
no added measurement noise, and quadrature spacings 0.01, 0.005, and 0.001.
All matrices have numerical rank 11. The figure uses spacing 0.01,
linear y-axes with scientific notation, and no percentile guides.

| Percentage | Condition bound | Relative coefficient-error bound |
|---:|---:|---:|
| 50 | 76.29 | 4.619e-9 |
| 90 | 386 | 3.199e-8 |
| 95 | 508.5 | 3.703e-8 |
| 99 | 655.5 | 5.821e-8 |
| 100 | 778.8 | 6.602e-8 |

Bounds are rounded upward to four significant figures and represent
separate marginal coverage of the observed sample. At percentage p > 0,
the curve gives order statistic ceil(n*p/100); at zero it gives the minimum.
No population or uniform stability guarantee is implied.

At spacing 0.001 the maximum coefficient error is 4.109e-11 and the largest
relative condition-number change from the primary grid is 6.45e-8.
The quadrature refinement keeps each dense numerical trajectory fixed.

The PNG contains the figure; `samples.csv` and `duffing_T100_samples.npz`
contain all results; `coverage_bounds.csv` includes marginal and joint
counts; `summary.json` records parameters, refinement results, and hashes.
