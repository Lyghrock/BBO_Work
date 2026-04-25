# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import numpy as np

def from_unit_cube(point, lb, ub):
    assert np.all(lb < ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert point.ndim  == 2
    new_point = point * (ub - lb) + lb
    return new_point

def latin_hypercube(n, dims):
    """Latin Hypercube Sampling - fully vectorized"""
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(n)) / (2.0 * n)
    for i in range(dims):
        points[:, i] = centers[np.random.permutation(n)]
    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) / (2.0 * n)
    points += perturbation
    np.clip(points, 0.0, 1.0, out=points)
    return points

