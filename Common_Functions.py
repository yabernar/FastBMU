import numpy as np


def quadratic_distance(a, b):
    assert np.array_equal(np.array(a.shape), np.array(b.shape))
    return np.sum((a - b) ** 2)


def manhattan_distance(a, b):
    return np.sum(np.abs(np.array(a) - np.array(b)))


def toroidal_distance(a, b, dimensions):
    direct = np.abs(np.array(a) - np.array(b))
    indirect = dimensions - direct
    return np.sum(np.minimum(direct, indirect))


# Taken from https://www.redblobgames.com/grids/hexagons/
def oddr_to_cube(hex):
    x = hex[0] - (hex[1] - (hex[1]&1)) // 2
    z = hex[1]
    y = -x-z
    return x, y, z


def cube_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def hexagonal_distance(x, y):
    a = oddr_to_cube(x)
    b = oddr_to_cube(y)
    return cube_distance(a, b)


def gaussian(d, sigma):
    return np.exp(-((d / sigma) ** 2) / 2) / sigma
    # Between 0 and 1/sig


def normalized_gaussian(d, sigma):
    return np.exp(-((d / sigma) ** 2) / 2)
    # Between 0 and 1

