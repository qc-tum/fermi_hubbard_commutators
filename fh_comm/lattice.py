import math
from itertools import product
from collections.abc import Sequence
import numpy as np


def periodic_wrap(c: Sequence[int], shape: Sequence[int]) -> tuple:
    """
    Periodic wrapping of coordinate `c` due to periodic boundary conditions
    for a rectangular unit cell of dimension `shape`.
    """
    return tuple(c[i] % shape[i] for i in range(len(shape)))


def index_to_latt_coord(i: int, shape: Sequence[int]) -> tuple:
    """
    Map linear index to integer lattice coordinate.
    """
    assert i < math.prod(shape)
    return np.unravel_index(i, shape)


def latt_coord_to_index(c: Sequence[int], shape: Sequence[int]) -> int:
    """
    Map integer lattice coordinate to linear index,
    assuming periodic boundary conditions.
    """
    return int(np.ravel_multi_index(periodic_wrap(c, shape), shape))


class SubLattice:
    """
    Sub-lattice specified by integer-valued basis vectors.
    """
    def __init__(self, basis: np.ndarray):
        basis = np.array(basis, dtype=int, copy=True)
        if basis.ndim != 2:
            raise ValueError("expecting a matrix for `basis`")
        nv = basis.shape[1]
        if np.linalg.matrix_rank(basis) != nv:
            raise ValueError("`basis` does not have full rank")
        # "optimize" basis, linear combination of basis vectors with smaller norm
        changed = True
        while changed:
            changed = False
            for i in range(basis.shape[1]):
                for j in range(basis.shape[1]):
                    if j == i:
                        continue
                    c = basis.copy()
                    c[:, i] -= c[:, j]
                    if np.linalg.norm(c, "fro") < np.linalg.norm(basis, "fro"):
                        basis = c
                        changed = True
                    c = basis.copy()
                    c[:, i] += c[:, j]
                    if np.linalg.norm(c, "fro") < np.linalg.norm(basis, "fro"):
                        basis = c
                        changed = True
        self.basis = basis

    def nearest_point(self, v):
        """
        Determine the sub-lattice point closest to `v`.
        """
        x = np.linalg.lstsq(self.basis, v, rcond=None)[0]
        x = np.array(np.around(x), dtype=int)
        return tuple(self.basis @ x)

    def nearest_center(self, vlist):
        """
        Determine the sub-lattice point minimizing the squared distances to the points in `vlist`.
        """
        vlist = np.asarray(vlist)
        # use nearest point to geometric center as initial guess
        c = self.nearest_point(np.mean(vlist, axis=0))
        d_min = None
        w_min = None
        for x in product((-1, 0, 1), repeat=self.basis.shape[1]):
            w = tuple(c + self.basis @ np.array(x))
            # not using np.norm here to retain integers
            d = sum(np.dot(v - w, v - w) for v in vlist)
            # use lexicographic order if otherwise equal
            if d_min is None or d < d_min or (d == d_min and w < w_min):
                d_min = d
                w_min = w
        return w_min

    def contains(self, v) -> bool:
        """
        Determine whether `v` is a point of the sub-lattice.
        """
        w = self.nearest_point(v)
        return np.array_equal(v, w)

    def instantiate(self, cmin: Sequence[int], cmax: Sequence[int]):
        """
        Generate the list of points belonging to the sublattice
        within the region defined by `cmin` (inclusive) and `cmax` (exclusive).
        """
        assert len(cmin) == self.basis.shape[0]
        assert len(cmax) == self.basis.shape[0]
        m = max(abs(c) for c in (list(cmin) + list(cmax)))
        ptlist = []
        for i in product(range(-m, m + 1), repeat=self.basis.shape[1]):
            pt = self.basis @ np.array(i)
            if all(cmin[k] <= pt[k] < cmax[k] for k in range(len(pt))):
                ptlist.append(tuple(pt))
        return ptlist
