"""
Distances between arrays of points on the plane.

Every function here takes a(M, D) and b(N, D) and returns an (M, N) matrix of scalars, so that
`metric(points[:, None], nodes[None, :])` gives every distance from every point to every node --
which is what a scattered-sample estimator needs and the only thing it needs.

Euclidean and nothing else, deliberately. A field over the plane is measured with a plane's ruler,
and anything that wants a different one -- distances on a sphere, say -- passes it in: every
corrector here takes `metric` as an argument. demeteor keeps the spherical one, because spherical
geometry is sky geometry and that is what demeteor is about.
"""
import numpy as np
from numpy.typing import NDArray


def euclidean(a: NDArray, b: NDArray) -> NDArray:
    """ Straight-line distance from every point of `a` to every point of `b`. """
    return np.sqrt(np.sum((a - b) ** 2, axis=2))
