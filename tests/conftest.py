import sys
import pytest
import numpy as np

sys.path.append('src/')

from demetria import ScalarField, VectorField


@pytest.fixture
def unit_square():
    x = np.linspace(-1, 1, 401)
    return np.meshgrid(x, x)


@pytest.fixture
def fine_unit_square():
    """
    The same square, resolved finely enough for a high-order polynomial.

    These norms are midpoint sums over a grid, and that quadrature has to resolve the oscillation it
    is integrating: at 401 points the error grows steadily with order -- 4e-4 at order 2, 4e-3 at
    order 11, 9e-3 at order 23 -- while at a fixed order it falls as the grid refines, 0.991173 at
    401 points, 0.997902 at 801, 0.999291 at 1601. So it is the grid and not the polynomial, and
    anything much above order ten wants this instead.
    """
    x = np.linspace(-1, 1, 1601)
    return np.meshgrid(x, x)

@pytest.fixture
def scalar_zero():
    return ScalarField()

@pytest.fixture
def radial():
    return ScalarField(lambda x, y: x + y)

@pytest.fixture
def trough():
    return ScalarField(lambda x, y: x**2 + y)

@pytest.fixture
def rotating_disk():
    return VectorField.from_function(lambda x, y: (y, -x))

@pytest.fixture
def source():
    return ScalarField(lambda x, y: (x, y))

