import numpy as np
import unittest
import fh_comm as fhc


class TestLattice(unittest.TestCase):

    def test_unit_cell(self):
        """
        Test coordinate translation to unit cell due to periodic boundary conditions.
        """
        rng = np.random.default_rng()
        for Lx in range(3, 5):
            for Ly in range(3, 5):
                ps = (Lx, Ly)
                c = (rng.integers(Lx), rng.integers(Ly))
                tx, ty = rng.integers(-2, 3, size=2)
                self.assertTrue(fhc.periodic_wrap((c[0] + tx*Lx, c[1] + ty*Ly), ps) == c)

    def test_lattice_coords(self):
        """
        Test lattice coordinate indexing.
        """
        for Lx in range(3, 5):
            for Ly in range(3, 5):
                ps = (Lx, Ly)
                for i in range(Lx*Ly):
                    self.assertEqual(i, fhc.latt_coord_to_index(
                                        fhc.index_to_latt_coord(i, ps), ps))

    def test_sublattice(self):
        """
        Test sub-lattice functionality.
        """
        rng = np.random.default_rng()
        for sdim in range(1, 4):
            for bdim in range(1, sdim + 1):
                # construct sub-lattice
                basis = np.zeros((sdim, bdim), dtype=int)
                while np.linalg.matrix_rank(basis) < bdim:
                    basis = rng.integers(-5, 6, size=(sdim, bdim))
                sublatt = fhc.SubLattice(basis)
                # nearest point
                v = rng.integers(-16, 17, size=sdim)
                vn = sublatt.nearest_point(v)
                x = np.linalg.lstsq(sublatt.basis, v - vn, rcond=None)[0]
                self.assertTrue(np.linalg.norm(x, np.inf) <= 0.5 + 10*np.finfo(float).eps)
                # determining whether point is contained in sub-lattice
                self.assertTrue(sublatt.contains(sublatt.basis @ rng.integers(-2, 3, size=bdim)))
                ptlist = sublatt.instantiate([-10, -13, -7, -11][:sdim], [12, 29, 17, 14][:sdim])
                for pt in ptlist:
                    self.assertTrue(sublatt.contains(pt))
        # nearest center
        sublatt = fhc.SubLattice(np.array([(3, 0, -3), (0, 3, -3)]).T)
        ptlist = np.array([( 0,  0,  0), (-3, -3,  6)])
        c = sublatt.nearest_center(ptlist)
        self.assertTrue(sublatt.contains(c))
        d_ref = sum(np.dot(pt - c, pt - c) for pt in ptlist)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                w = i*sublatt.basis[:, 0] + j*sublatt.basis[:, 1]
                d = sum(np.dot(pt - w, pt - w) for pt in ptlist)
                self.assertTrue(d_ref <= d)


if __name__ == "__main__":
    unittest.main()
