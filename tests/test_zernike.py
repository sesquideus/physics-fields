import sys
import pytest
import numpy as np

from demetria import Field, ScalarField, VectorField
from demetria.zernike import Zernike, ZernikeVector


nl = [(n, l) for n in range(0, 10) for l in range(-n, n + 1, 2)]
nl2 = [(*a, *b) for a in nl for b in nl]


class TestZernikeScalar():
    def test_create(self):
        assert isinstance(Zernike(0, 0), ScalarField)

    def test_create_3m3(self):
        assert isinstance(Zernike(3, -3), ScalarField)

    def test_create_wrong_n(self):
        with pytest.raises(ValueError):
            _ = Zernike(-2, 2)

    def test_create_wrong_l(self):
        with pytest.raises(ValueError):
            _ = Zernike(3, -2)

    @staticmethod
    def eval(v1, v2, field):
        x, y = field
        z = np.ma.masked_where(Field.UnitDiskMask(x, y), (v1 * v2)(x, y))
        return np.sum(z) / z.count()

    @pytest.mark.parametrize("n, l", nl)
    def test_norm(self, n, l, unit_square):
        v = Zernike(n, l)
        assert self.eval(v, v, unit_square) == pytest.approx(1, rel=0.01)

    @pytest.mark.parametrize("n1, l1, n2, l2", nl2)
    def test_norm_x(self, n1, l1, n2, l2, unit_square):
        v1 = Zernike(n1, l1)
        v2 = Zernike(n2, l2)
        assert self.eval(v1, v2, unit_square) == (pytest.approx(1, rel=0.01) if n1 == n2 and l1 == l2 else pytest.approx(0, abs=0.01))



class TestZernikeVector():
    def test_create_wrong_n(self):
        with pytest.raises(ValueError):
            _ = ZernikeVector(-2, 7)

    def test_create_wrong_l(self):
        with pytest.raises(ValueError):
            _ = ZernikeVector(3, -2)

    def test_create_wrong_laplacian(self):
        with pytest.raises(ValueError):
            _ = ZernikeVector(4, 4, False)

    def test_create_wrong_nonlaplacian(self):
        with pytest.raises(ValueError):
            _ = ZernikeVector(4, 0)

    @staticmethod
    def eval(v1, v2, field):
        x, y = field
        mask = Field.UnitDiskMask(x, y)
        mask = np.stack([mask, mask])
        z = np.ma.masked_where(mask, (v1 * v2)(x, y))
        return 2 * np.sum(z) / z.count()

    def test_ortho_1(self, unit_square):
        v1 = ZernikeVector(3, -1, False)
        v2 = ZernikeVector(5, 3, True)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_ortho_2(self, unit_square):
        v1 = ZernikeVector(1, -1)
        v2 = ZernikeVector(5, 5)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_ortho_3(self, unit_square):
        v1 = ZernikeVector(6, 4, True)
        v2 = ZernikeVector(8, -2, True)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_ortho_4(self, unit_square):
        v1 = ZernikeVector(2, 0, True)
        v2 = ZernikeVector(4, 0, False)
        assert self.eval(v1, v2, unit_square) == pytest.approx(0, abs=0.002)

    def test_norm_1(self, unit_square):
        v1 = ZernikeVector(2, 0, True)
        assert self.eval(v1, v1, unit_square) == pytest.approx(1, abs=0.005)

    def test_norm_2(self, unit_square):
        v1 = ZernikeVector(4, 2, False)
        assert self.eval(v1, v1, unit_square) == pytest.approx(1, abs=0.005)

    def test_norm_3(self, unit_square):
        v1 = ZernikeVector(5, 5)
        assert self.eval(v1, v1, unit_square) == pytest.approx(1, abs=0.005)

    def test_norm_4(self, fine_unit_square):
        """ Order 23, so on the fine grid -- see the fixture for why, and for the numbers. """
        v1 = ZernikeVector(23, 17, False)
        assert self.eval(v1, v1, fine_unit_square) == pytest.approx(1, abs=0.005)
