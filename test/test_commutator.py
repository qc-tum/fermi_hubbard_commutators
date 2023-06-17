import unittest
import scipy.sparse.linalg as spla
import fh_comm as fhc


class TestCommutator(unittest.TestCase):

    def test_hopping_commutator_1d(self):
        """
        Test commutation relations for hopping operators on a 1D lattice.
        """
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        for x in range(L):
            for y in range(L):
                if y == x:
                    continue
                for s in [0, 1]:
                    ha = fhc.HoppingOp((1,), (3,), 1,  0.4)
                    hb = fhc.HoppingOp((x,), (y,), s, -0.7)
                    hc = fhc.commutator(ha, hb)
                    ha_mat = ha.as_field_operator().as_matrix((L,))
                    hb_mat = hb.as_field_operator().as_matrix((L,))
                    hc_mat = hc.as_field_operator().as_matrix((L,))
                    self.assertEqual(spla.norm(hc_mat - comm(ha_mat, hb_mat)), 0)
                    # include lattice translations
                    hct = fhc.commutator_translation(ha, hb, translatt)
                    hat_mat =  ha.as_field_operator().as_matrix((L,), translatt)
                    hbt_mat =  hb.as_field_operator().as_matrix((L,), translatt)
                    hct_mat = hct.as_field_operator().as_matrix((L,), translatt)
                    self.assertEqual(spla.norm(hct_mat - comm(hat_mat, hbt_mat)), 0)

    def test_hopping_commutator_2d(self):
        """
        Test commutation relations for hopping operators on a 2D lattice.
        """
        latt_shape = (2, 2)
        for x in range(latt_shape[0]):
            for y in range(latt_shape[1]):
                for z in [0, 1]:
                    if (x, y) == (z, 1):
                        continue
                    for s in [0, 1]:
                        ha = fhc.HoppingOp((0, 1), (1, 0), 1,  0.4)
                        hb = fhc.HoppingOp((x, y), (z, 1), s, -0.7)
                        hc = fhc.commutator(ha, hb)
                        ha_mat = ha.as_field_operator().as_matrix(latt_shape)
                        hb_mat = hb.as_field_operator().as_matrix(latt_shape)
                        hc_mat = hc.as_field_operator().as_matrix(latt_shape)
                        self.assertEqual(spla.norm(hc_mat - comm(ha_mat, hb_mat)), 0)

    def test_mixed_symm_hopping_commutator_1d(self):
        """
        Test commutation relations between a symmetric and
        an anti-symmetric hopping operator on a 1D lattice.
        """
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        for x in range(L):
            for y in range(L):
                if y == x:
                    continue
                for s in [0, 1]:
                    ha = fhc.HoppingOp((1,), (3,), 1, 0.4)
                    hb = fhc.AntisymmHoppingOp((x,), (y,), s, -0.7)
                    hc = fhc.commutator(ha, hb)
                    ha_mat = ha.as_field_operator().as_matrix((L,))
                    hb_mat = hb.as_field_operator().as_matrix((L,))
                    hc_mat = hc.as_field_operator().as_matrix((L,))
                    self.assertEqual(spla.norm(hc_mat - comm(ha_mat, hb_mat)), 0)
                    # include lattice translations
                    hct = fhc.commutator_translation(ha, hb, translatt)
                    hat_mat =  ha.as_field_operator().as_matrix((L,), translatt)
                    hbt_mat =  hb.as_field_operator().as_matrix((L,), translatt)
                    hct_mat = hct.as_field_operator().as_matrix((L,), translatt)
                    self.assertEqual(spla.norm(hct_mat - comm(hat_mat, hbt_mat)), 0)

    def test_mixed_symm_hopping_commutator_2d(self):
        """
        Test commutation relations between a symmetric and
        an anti-symmetric hopping operator on a 2D lattice.
        """
        latt_shape = (2, 2)
        for x in range(latt_shape[0]):
            for y in range(latt_shape[1]):
                for z in [0, 1]:
                    if (x, y) == (z, 1):
                        continue
                    for s in [0, 1]:
                        ha = fhc.AntisymmHoppingOp((0, 1), (1, 0), 1,  0.4)
                        hb = fhc.HoppingOp((x, y), (z, 1), s, -0.7)
                        hc = fhc.commutator(ha, hb)
                        ha_mat = ha.as_field_operator().as_matrix(latt_shape)
                        hb_mat = hb.as_field_operator().as_matrix(latt_shape)
                        hc_mat = hc.as_field_operator().as_matrix(latt_shape)
                        self.assertEqual(spla.norm(hc_mat - comm(ha_mat, hb_mat)), 0)

    def test_antisymm_hopping_commutator_1d(self):
        """
        Test commutation relations for anti-symmetric hopping operators on a 1D lattice.
        """
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        for x in range(L):
            for y in range(L):
                if y == x:
                    continue
                for s in [0, 1]:
                    ha = fhc.AntisymmHoppingOp((3,), (2,), 1,  0.4)
                    hb = fhc.AntisymmHoppingOp((x,), (y,), s, -0.7)
                    hc = fhc.commutator(ha, hb)
                    ha_mat = ha.as_field_operator().as_matrix((L,))
                    hb_mat = hb.as_field_operator().as_matrix((L,))
                    hc_mat = hc.as_field_operator().as_matrix((L,))
                    self.assertEqual(spla.norm(hc_mat - comm(ha_mat, hb_mat)), 0)
                    # include lattice translations
                    hct = fhc.commutator_translation(ha, hb, translatt)
                    hat_mat =  ha.as_field_operator().as_matrix((L,), translatt)
                    hbt_mat =  hb.as_field_operator().as_matrix((L,), translatt)
                    hct_mat = hct.as_field_operator().as_matrix((L,), translatt)
                    self.assertEqual(spla.norm(hct_mat - comm(hat_mat, hbt_mat)), 0)

    def test_antisymm_hopping_commutator_2d(self):
        """
        Test commutation relations for anti-symmetric hopping operators on a 2D lattice.
        """
        latt_shape = (2, 2)
        for x in range(latt_shape[0]):
            for y in range(latt_shape[1]):
                for z in [0, 1]:
                    if (x, y) == (z, 1):
                        continue
                    for s in [0, 1]:
                        ha = fhc.AntisymmHoppingOp((0, 1), (1, 0), 1,  0.4)
                        hb = fhc.AntisymmHoppingOp((x, y), (z, 1), s, -0.7)
                        hc = fhc.commutator(ha, hb)
                        ha_mat = ha.as_field_operator().as_matrix(latt_shape)
                        hb_mat = hb.as_field_operator().as_matrix(latt_shape)
                        hc_mat = hc.as_field_operator().as_matrix(latt_shape)
                        self.assertEqual(spla.norm(hc_mat - comm(ha_mat, hb_mat)), 0)

    def test_hopping_number_commutator(self):
        """
        Test commutation relations between a hopping and number operator.
        """
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        for x in range(L):
            for y in range(L):
                if y == x:
                    continue
                for s in [0, 1]:
                    ha = fhc.NumberOp((2,), 1,  0.4)
                    hb = fhc.HoppingOp((x,), (y,), s, -0.7)
                    hc = fhc.commutator(ha, hb)
                    ha_mat = ha.as_field_operator().as_matrix((L,))
                    hb_mat = hb.as_field_operator().as_matrix((L,))
                    hc_mat = hc.as_field_operator().as_matrix((L,))
                    self.assertEqual(spla.norm(hc_mat - comm(ha_mat, hb_mat)), 0)
                    # include lattice translations
                    hct = fhc.commutator_translation(ha, hb, translatt)
                    hat_mat =  ha.as_field_operator().as_matrix((L,), translatt)
                    hbt_mat =  hb.as_field_operator().as_matrix((L,), translatt)
                    hct_mat = hct.as_field_operator().as_matrix((L,), translatt)
                    self.assertEqual(spla.norm(hct_mat - comm(hat_mat, hbt_mat)), 0)

    def test_antisymm_hopping_number_commutator(self):
        """
        Test commutation relations between an anti-symmetric hopping and number operator.
        """
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        for x in range(L):
            for y in range(L):
                if y == x:
                    continue
                for s in [0, 1]:
                    ha = fhc.NumberOp((2,), 1,  0.4)
                    hb = fhc.AntisymmHoppingOp((x,), (y,), s, -0.7)
                    hc = fhc.commutator(ha, hb)
                    ha_mat = ha.as_field_operator().as_matrix((L,))
                    hb_mat = hb.as_field_operator().as_matrix((L,))
                    hc_mat = hc.as_field_operator().as_matrix((L,))
                    self.assertEqual(spla.norm(hc_mat - comm(ha_mat, hb_mat)), 0)
                    # include lattice translations
                    hct = fhc.commutator_translation(ha, hb, translatt)
                    hat_mat =  ha.as_field_operator().as_matrix((L,), translatt)
                    hbt_mat =  hb.as_field_operator().as_matrix((L,), translatt)
                    hct_mat = hct.as_field_operator().as_matrix((L,), translatt)
                    self.assertEqual(spla.norm(hct_mat - comm(hat_mat, hbt_mat)), 0)

    def test_product_commutator(self):
        """
        Test commutator with a product operator.
        """
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        for x in range(L):
            for y in range(L):
                if y == x:
                    continue
                for s in [0, 1]:
                    ha = fhc.AntisymmHoppingOp((x,), (y,), s, -0.7)
                    hb = fhc.ProductOp([fhc.NumberOp((3,), 0, 0.5), fhc.AntisymmHoppingOp((1,), (3,), 1, 0.2), fhc.HoppingOp((0,), (3,), 0, 1.3)], 0.7)
                    hc = fhc.commutator(ha, hb)
                    ha_mat = ha.as_field_operator().as_matrix((L,))
                    hb_mat = hb.as_field_operator().as_matrix((L,))
                    hc_mat = hc.as_field_operator().as_matrix((L,))
                    self.assertEqual(spla.norm(hc_mat - comm(ha_mat, hb_mat)), 0)
                    # interchange ha <-> hb
                    hc = fhc.commutator(hb, ha)
                    hc_mat = hc.as_field_operator().as_matrix((L,))
                    self.assertEqual(spla.norm(hc_mat - comm(hb_mat, ha_mat)), 0)
                    # include lattice translations
                    hct = fhc.commutator_translation(ha, hb, translatt)
                    hat_mat =  ha.as_field_operator().as_matrix((L,), translatt)
                    hbt_mat =  hb.as_field_operator().as_matrix((L,), translatt)
                    hct_mat = hct.as_field_operator().as_matrix((L,), translatt)
                    self.assertAlmostEqual(spla.norm(hct_mat - comm(hat_mat, hbt_mat)), 0)
                    # interchange ha <-> hb
                    hct = fhc.commutator_translation(hb, ha, translatt)
                    hct_mat = hct.as_field_operator().as_matrix((L,), translatt)
                    self.assertAlmostEqual(spla.norm(hct_mat - comm(hbt_mat, hat_mat)), 0)

    def test_sum_commutator(self):
        """
        Test commutator with a summation operator.
        """
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        for x in range(L):
            for y in range(L):
                if y == x:
                    continue
                for s in [0, 1]:
                    ha = fhc.ProductOp([fhc.NumberOp((2,), 0, 0.6), fhc.AntisymmHoppingOp((x,), (y,), s, -0.7)], 1.1)
                    hb = fhc.SumOp([
                        fhc.ProductOp([fhc.NumberOp((3,), 0, 0.5), fhc.AntisymmHoppingOp((1,), (3,), 1, 0.2)], 0.7),
                        fhc.ZeroOp(), fhc.HoppingOp((0,), (3,), 0, 1.3), fhc.NumberOp((1,), 1, 0.4)])
                    hc = fhc.commutator(ha, hb)
                    ha_mat = ha.as_field_operator().as_matrix((L,))
                    hb_mat = hb.as_field_operator().as_matrix((L,))
                    hc_mat = hc.as_field_operator().as_matrix((L,))
                    self.assertAlmostEqual(spla.norm(hc_mat - comm(ha_mat, hb_mat)), 0)
                    # interchange ha <-> hb
                    hc = fhc.commutator(hb, ha)
                    hc_mat = hc.as_field_operator().as_matrix((L,))
                    self.assertAlmostEqual(spla.norm(hc_mat - comm(hb_mat, ha_mat)), 0)
                    # include lattice translations
                    hct = fhc.commutator_translation(ha, hb, translatt)
                    hat_mat =  ha.as_field_operator().as_matrix((L,), translatt)
                    hbt_mat =  hb.as_field_operator().as_matrix((L,), translatt)
                    hct_mat = hct.as_field_operator().as_matrix((L,), translatt)
                    self.assertAlmostEqual(spla.norm(hct_mat - comm(hat_mat, hbt_mat)), 0)
                    # interchange ha <-> hb
                    hct = fhc.commutator_translation(hb, ha, translatt)
                    hct_mat = hct.as_field_operator().as_matrix((L,), translatt)
                    self.assertAlmostEqual(spla.norm(hct_mat - comm(hbt_mat, hat_mat)), 0)


def comm(a, b):
    """
    Commutator [a, b] = a b - b a.
    """
    return a @ b - b @ a


if __name__ == "__main__":
    unittest.main()
