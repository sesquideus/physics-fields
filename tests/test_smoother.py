"""
The kernel smoother itself, which had no tests at all before it moved here.

It is a Nadaraya-Watson estimator: the value at a node is the mean of the samples, each weighted by
a kernel of its distance over the bandwidth. Everything worth asserting about it follows from that
one sentence, and the two ends of the bandwidth are where it is easiest to be sure -- at a small
one it reproduces the nearest sample, at a large one it reproduces the global mean, and both are
exactly computable.
"""
import numpy as np
import pytest

from demetria.correctors import Interpolator, KernelSmoother, kernels
from demetria.metrics import euclidean


@pytest.fixture
def points():
    """ A small square lattice, so that "the nearest sample" is unambiguous. """
    s = np.linspace(-1, 1, 5)
    x, y = np.meshgrid(s, s)
    return np.stack([x.ravel(), y.ravel()], axis=1)


@pytest.fixture
def scalars(points):
    """ One component, as a magnitude correction has. """
    return (points[:, :1] * 2.0) + 1.0


@pytest.fixture
def vectors(points):
    """ Two, as a position correction has. """
    return np.stack([points[:, 0] * 2.0, points[:, 1] * -3.0], axis=1)


class TestTheTwoLimits:
    def test_a_tiny_bandwidth_reproduces_the_nearest_sample(self, points, vectors):
        """
        As the bandwidth goes to zero the weights collapse onto the closest point, so evaluating at
        a sample returns that sample. Not a useful field -- it says nothing between the points --
        but it is the limit, and it is what makes in-sample error useless for choosing a bandwidth.
        """
        smoother = KernelSmoother(points, vectors, bandwidth=1e-4)

        assert np.allclose(smoother(points), vectors, atol=1e-6)

    def test_a_huge_bandwidth_reproduces_the_global_mean(self, points, vectors):
        """ Every weight equal, so every node gets the same answer: the mean of the samples. """
        smoother = KernelSmoother(points, vectors, bandwidth=1e6)
        nodes = np.array([[0.0, 0.0], [0.5, -0.5], [-0.9, 0.1]])

        assert np.allclose(smoother(nodes), vectors.mean(axis=0), atol=1e-3)


class TestWhatItGivesBack:
    def test_the_shape_follows_the_values(self, points, scalars, vectors):
        nodes = np.array([[0.1, 0.2], [-0.3, 0.4], [0.0, 0.0]])

        assert KernelSmoother(points, scalars, bandwidth=0.3)(nodes).shape == (3, 1)
        assert KernelSmoother(points, vectors, bandwidth=0.3)(nodes).shape == (3, 2)

    def test_it_stays_inside_the_range_of_its_samples(self, points, vectors):
        """
        A weighted mean cannot leave the convex hull of what it averages, whatever the bandwidth.
        Worth pinning because it is the property that makes a smoother safe where an interpolating
        polynomial is not.
        """
        nodes = np.random.default_rng(20260825).uniform(-1, 1, (200, 2))

        for bandwidth in (0.01, 0.1, 1.0, 10.0):
            estimate = KernelSmoother(points, vectors, bandwidth=bandwidth)(nodes)

            assert np.all(estimate >= vectors.min(axis=0) - 1e-9)
            assert np.all(estimate <= vectors.max(axis=0) + 1e-9)

    def test_a_node_far_outside_still_gets_an_answer(self, points, vectors):
        """
        nexp has infinite support, so no node is ever left with no neighbours -- it just gets the
        mean of the nearest few. A compactly supported kernel would divide by zero out here.
        """
        estimate = KernelSmoother(points, vectors, bandwidth=0.2)(np.array([[50.0, 50.0]]))

        assert np.all(np.isfinite(estimate))


class TestTheKernelIsTheCharacter:
    def test_a_different_kernel_gives_a_different_field(self, points, vectors):
        nodes = np.array([[0.25, 0.25]])
        nexp = KernelSmoother(points, vectors, bandwidth=0.3, kernel=kernels.nexp)(nodes)
        gauss = KernelSmoother(points, vectors, bandwidth=0.3, kernel=kernels.ugauss)(nodes)

        assert not np.allclose(nexp, gauss)

    def test_epanechnikov_has_compact_support_and_says_so(self, points, vectors):
        """
        Kept as a warning as much as an option: it is zero beyond one bandwidth, so a node with no
        sample within one gets 0/0. That is the kernel's nature and not a fault, but it is why nexp
        is the default.
        """
        smoother = KernelSmoother(points, vectors, bandwidth=0.01,
                                  kernel=kernels.epanechnikov)

        with np.errstate(invalid='ignore', divide='ignore'):
            assert not np.all(np.isfinite(smoother(np.array([[9.0, 9.0]]))))

    def test_the_metric_is_injectable(self, points, vectors):
        """ Which is how a caller uses a spherical distance without this package knowing one. """
        calls = []

        def counted(a, b):
            calls.append(1)
            return euclidean(a, b)

        KernelSmoother(points, vectors, bandwidth=0.3, metric=counted)(points)

        assert calls


class TestEmptyAndDegenerate:
    def test_no_samples_at_all_gives_zeros_and_not_a_nan(self):
        """
        There is nothing to average, so the honest answers are a nan or a refusal; it returns zeros
        instead. Pinned because it is a real caller's real case -- a frame in which no reference
        star was matched -- and because a zero correction is at least a correction of nothing,
        whereas a nan would poison whatever it was added to.
        """
        smoother = KernelSmoother(np.empty((0, 2)), np.empty((0, 2)), bandwidth=0.1)
        estimate = smoother(np.array([[0.0, 0.0], [1.0, 1.0]]))

        assert estimate.shape == (2, 2), "one row per node, and the right width"
        assert np.all(estimate == 0.0)

    def test_one_sample_is_a_constant_field(self):
        smoother = KernelSmoother(np.array([[0.0, 0.0]]), np.array([[7.0, -3.0]]), bandwidth=0.5)
        nodes = np.array([[0.0, 0.0], [1.0, 1.0], [-2.0, 3.0]])

        assert np.allclose(smoother(nodes), [7.0, -3.0])


class TestInterpolator:
    def test_it_reproduces_its_samples(self, points, scalars):
        """ Unlike the smoother, an interpolator is exact at the data by construction. """
        estimate = Interpolator(points, scalars, method='linear')(points)

        assert np.allclose(estimate, scalars, atol=1e-9)

    def test_outside_the_hull_it_says_it_does_not_know(self, points, scalars):
        """ nan rather than an extrapolation, which is the honest answer and the fill_value. """
        estimate = Interpolator(points, scalars, method='linear')(np.array([[9.0, 9.0]]))

        assert np.all(np.isnan(estimate))
