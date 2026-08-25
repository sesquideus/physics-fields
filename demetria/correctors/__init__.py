"""
Fields fitted to scattered samples.

The rest of demetria describes a field by a formula (`ScalarField`, `VectorField`), by values on a
grid (`SampledScalarField`) or by a basis expansion (`Zernike`). This is the fourth way: you have
measurements at N places that are nowhere in particular, and you want the value somewhere else.

A corrector takes (points, values) and gives back something callable. What it does between the
points is the whole of its character -- `KernelSmoother` takes a weighted mean of the neighbours,
`Interpolator` triangulates -- and for the smoother that character is governed by one number, the
bandwidth, which `bandwidth.select` chooses by cross-validation rather than leaving to a guess.

Named for what they were built for: correcting an all-sky plate, where the residual left over after
a twelve-parameter fit is a smooth field over the sensor that no twelve parameters can express.
Nothing here knows that, and none of it is about meteors.
"""
from . import bandwidth, kernels
from .base import BaseCorrector
from .interpolator import Interpolator
from .kernelsmoother import KernelSmoother

__all__ = ['BaseCorrector', 'Interpolator', 'KernelSmoother', 'bandwidth', 'kernels']
