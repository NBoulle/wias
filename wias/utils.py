"""Small utility helpers shared by examples."""

from __future__ import annotations


def format_equation(component_name, names, coeffs, min_magnitude=1e-2):
    """Format a sparse polynomial model as a readable equation string."""
    terms = []
    for name, coeff in zip(names, coeffs):
        if abs(coeff) > min_magnitude:
            terms.append("{:.6f}*{}".format(coeff, name))

    if not terms:
        return "{}' = 0".format(component_name)

    return "{}' = {}".format(component_name, " + ".join(terms))
