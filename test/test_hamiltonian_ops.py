from fractions import Fraction
import unittest
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import fh_comm as fhc


class TestHamiltonianOps(unittest.TestCase):

    def test_hopping_op(self):
        """
        Test hopping operator functionality.
        """
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        coeff = 0.7
        for x in range(L):
            for s in [0, 1]:
                h = fhc.HoppingOp((x,), (x + 1,), s, coeff)
                self.assertEqual(h.support(), [(x, s), (x + 1, s)])
                self.assertEqual(h.fermi_weight, 2)
                h2 = fhc.HoppingOp((x,), (x + 1,), s, Fraction(2, 3))
                self.assertTrue(h.proportional(h2))
                self.assertFalse(h == h2)
                self.assertTrue(h2 < h)
                self.assertFalse(h < h2)
                self.assertTrue(h < fhc.AntisymmHoppingOp((x,), (x + 1,), s, coeff))
                self.assertTrue(h + fhc.ZeroOp() == h)
                self.assertTrue(h - fhc.ZeroOp() == h)
                self.assertTrue(h + h2 == fhc.HoppingOp((x,), (x + 1,), s, coeff + Fraction(2, 3)))
                self.assertTrue(h - h2 == fhc.HoppingOp((x,), (x + 1,), s, coeff - Fraction(2, 3)))
                # eigenvalues of hopping operator (including spin)
                eighop = np.kron([0, 1, -1, 0], [1, 1, 1, 1])
                hmat = h.as_field_operator().as_matrix((L,))
                # must be Hermitian
                self.assertEqual(spla.norm(hmat.conj().T - hmat), 0)
                eig_ref = np.sort(np.kron(eighop, np.ones(16)).reshape(-1))
                self.assertTrue(np.allclose(np.linalg.eigvalsh(hmat.todense()), coeff * np.array(eig_ref)))
                self.assertEqual(h.norm_bound(), np.linalg.norm(hmat.todense(), ord=2))
                # include translated copies on sublattice
                hmat = h.as_field_operator().as_matrix((L,), translatt)
                # must be Hermitian
                self.assertEqual(spla.norm(hmat.conj().T - hmat), 0)
                # take shifted copy into account
                eig_ref = np.sort(np.add.outer(eighop, eighop).reshape(-1))
                self.assertTrue(np.allclose(np.linalg.eigvalsh(hmat.todense()), coeff * np.array(eig_ref)))
                ht = h.translate((-3,))
                self.assertEqual(ht.support(), [(x - 3, s), (x - 2, s)])

    def test_antisymm_hopping_op(self):
        """
        Test antisymmetrized hopping operator functionality.
        """
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        coeff = 0.7
        for x in range(L):
            for s in [0, 1]:
                h = fhc.AntisymmHoppingOp((x,), (x + 1,), s, coeff)
                self.assertEqual(h.support(), [(x, s), (x + 1, s)])
                self.assertEqual(h.fermi_weight, 2)
                h2 = fhc.AntisymmHoppingOp((x,), (x + 1,), s, 0.25)
                self.assertTrue(h.proportional(h2))
                self.assertFalse(h == h2)
                self.assertTrue(h2 < h)
                self.assertFalse(h < h2)
                self.assertTrue(fhc.HoppingOp((x,), (x + 1,), s, coeff) < h)
                self.assertTrue(h + fhc.ZeroOp() == h)
                self.assertTrue(h - fhc.ZeroOp() == h)
                self.assertTrue(h + h2 == fhc.AntisymmHoppingOp((x,), (x + 1,), s, coeff + 0.25))
                self.assertTrue(h - h2 == fhc.AntisymmHoppingOp((x,), (x + 1,), s, coeff - 0.25))
                # eigenvalues of anti-symmetric hopping operator (including spin),
                # without imaginary factor i
                eighop = np.kron([0, 1, -1, 0], [1, 1, 1, 1])
                hmat = h.as_field_operator().as_matrix((L,))
                # must be anti-Hermitian
                self.assertEqual(spla.norm(hmat.conj().T + hmat), 0)
                eig_ref = np.sort(np.kron(eighop, np.ones(16)).reshape(-1))
                self.assertTrue(np.allclose(np.linalg.eigvalsh(1j * hmat.todense()), coeff * np.array(eig_ref)))
                self.assertEqual(h.norm_bound(), np.linalg.norm(hmat.todense(), ord=2))
                # include translated copies on sublattice
                hmat = h.as_field_operator().as_matrix((L,), translatt)
                # must be anti-Hermitian
                self.assertEqual(spla.norm(hmat.conj().T + hmat), 0)
                # take shifted copy into account
                eig_ref = np.sort(np.add.outer(eighop, eighop).reshape(-1))
                self.assertTrue(np.allclose(np.linalg.eigvalsh(1j * hmat.todense()), coeff * np.array(eig_ref)))
                ht = h.translate((-3,))
                self.assertEqual(ht.support(), [(x - 3, s), (x - 2, s)])

    def test_number_op(self):
        """
        Test number operator functionality.
        """
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        coeff = 0.7
        for x in range(L):
            for s in [0, 1]:
                h = fhc.NumberOp((x,), s, coeff)
                self.assertEqual(h.support(), [(x, s)])
                self.assertEqual(h.fermi_weight, 2)
                h2 = fhc.NumberOp((x,), s, 0.25)
                self.assertTrue(h.proportional(h2))
                self.assertFalse(h == h2)
                self.assertTrue(h2 < h)
                self.assertFalse(h < h2)
                self.assertTrue(fhc.AntisymmHoppingOp((x,), (x + 1,), s, coeff) < h)
                self.assertTrue(h + fhc.ZeroOp() == h)
                self.assertTrue(h - fhc.ZeroOp() == h)
                self.assertTrue(h + h2 == fhc.NumberOp((x,), s, coeff + 0.25))
                self.assertTrue(h - h2 == fhc.NumberOp((x,), s, coeff - 0.25))
                # eigenvalues of number operator (for two sites, including spin)
                eignum = np.kron([0, 1], 8 * [1])
                hmat = h.as_field_operator().as_matrix((L,))
                # must be Hermitian
                self.assertEqual(spla.norm(hmat.conj().T - hmat), 0)
                eig_ref = np.sort(np.kron(eignum, np.ones(16)).reshape(-1))
                self.assertTrue(np.allclose(np.linalg.eigvalsh(hmat.todense()), coeff * np.array(eig_ref)))
                self.assertEqual(h.norm_bound(), np.linalg.norm(hmat.todense(), ord=2))
                # include translated copies on sublattice
                hmat = h.as_field_operator().as_matrix((L,), translatt)
                # must be Hermitian
                self.assertEqual(spla.norm(hmat.conj().T - hmat), 0)
                # take shifted copy into account
                eig_ref = np.sort(np.add.outer(eignum, eignum).reshape(-1))
                self.assertTrue(np.allclose(np.linalg.eigvalsh(hmat.todense()), coeff * np.array(eig_ref)))
                ht = h.translate((-3,))
                self.assertEqual(ht.support(), [(x - 3, s)])

    def test_zero_op(self):
        """
        Test zero operator functionality.
        """
        zop = fhc.ZeroOp()
        self.assertEqual(zop.support(), [])
        self.assertEqual(zop.fermi_weight, 0)
        self.assertTrue(zop.proportional(fhc.ZeroOp()))
        self.assertTrue(zop == fhc.ZeroOp())
        self.assertFalse(zop < fhc.ZeroOp())
        self.assertTrue(zop < fhc.HoppingOp((4,), (7,), 0, 0.2))
        self.assertEqual(zop.norm_bound(), 0)
        self.assertTrue(isinstance(-zop, fhc.ZeroOp))
        self.assertEqual(spla.norm(zop.as_field_operator().as_matrix((4,))), 0)

    def test_product_op(self):
        """
        Test product operator functionality.
        """
        # lattice size
        L = 4
        coeff = 0.7
        ops = [fhc.NumberOp((3,), 0, 0.5), fhc.ProductOp([fhc.AntisymmHoppingOp((1,), (3,), 1, 0.2), fhc.HoppingOp((0,), (3,), 0, 1.3)], 1.1)]
        prod = fhc.ProductOp(ops, coeff)
        # nested products must be flattened out
        self.assertEqual(len(prod.ops), 3)
        self.assertEqual(prod.support(), [(0, 0), (1, 1), (3, 0), (3, 1)])
        self.assertEqual(prod.fermi_weight, 6)
        self.assertFalse(prod.is_numop_product())
        ops2 = [fhc.NumberOp((3,), 0, 0.4), fhc.ProductOp([fhc.AntisymmHoppingOp((1,), (3,), 1, -0.3), fhc.HoppingOp((0,), (3,), 0, 0.6)], 1.7)]
        prod2 = fhc.ProductOp(ops2, 0.1)
        self.assertTrue(prod.proportional(prod2))
        self.assertFalse(prod == prod2)
        self.assertTrue(prod2 < prod)
        self.assertFalse(prod < prod2)
        self.assertTrue(fhc.HoppingOp((-1,), (2,), 1, 0.1) < prod)
        ops3 = [fhc.NumberOp((3,), 0, 0.5), fhc.ProductOp([fhc.HoppingOp((1,), (3,), 1, 0.2), fhc.HoppingOp((0,), (3,), 0, 1.3)], 1.1)]
        self.assertTrue(fhc.ProductOp(ops3, 0.1) < prod)
        self.assertTrue(prod + fhc.ZeroOp() == prod)
        self.assertTrue(prod - fhc.ZeroOp() == prod)
        self.assertTrue(prod + prod2 == fhc.ProductOp(prod2.ops, prod.coeff + prod2.coeff))
        self.assertTrue(prod - prod2 == fhc.ProductOp(prod2.ops, prod.coeff - prod2.coeff))
        prod_mat = prod.as_field_operator().as_matrix((L,))
        prod_ref = coeff * sparse.identity(2**(2*L))
        for op in ops:
            prod_ref = prod_ref @ op.as_field_operator().as_matrix((L,))
        self.assertEqual(spla.norm(prod_mat - prod_ref), 0)
        self.assertEqual(spla.norm(prod_mat + (-prod).as_field_operator().as_matrix((L,))), 0)
        self.assertAlmostEqual(prod.norm_bound(), np.linalg.norm(prod_mat.todense(), ord=2), delta=1e-12)
        prodt = prod.translate((-7,))
        self.assertEqual(prodt.support(), [(-7, 0), (-6, 1), (-4, 0), (-4, 1)])
        # multiply with zero operator
        prod = fhc.ProductOp(ops + [fhc.ZeroOp()], coeff)
        self.assertEqual(prod.support(), [])
        self.assertEqual(prod.norm_bound(), 0)
        self.assertEqual(spla.norm(prod.as_field_operator().as_matrix((L,))), 0)

    def test_sum_op(self):
        """
        Test sum operator functionality.
        """
        # lattice size
        L = 4
        terms = [fhc.ProductOp([fhc.NumberOp((3,), 0, 0.5), fhc.AntisymmHoppingOp((1,), (3,), 1, 0.2)], 1.7),
                 fhc.ZeroOp(),
                 fhc.SumOp([fhc.HoppingOp((0,), (3,), 0, 1.3), fhc.NumberOp((1,), 1, 0.4)])]
        h = fhc.SumOp(terms)
        self.assertEqual(h.support(), [(0, 0), (1, 1), (3, 0), (3, 1)])
        self.assertEqual(h.fermi_weight, 4)
        self.assertFalse(h.is_numop_sum())
        h2 = fhc.SumOp(reversed(terms))
        self.assertTrue(h.proportional(h2))
        self.assertTrue(h == h2)
        self.assertFalse(h < h2)
        self.assertFalse(h2 < h)
        # first term in `h` is a HoppingOp
        self.assertTrue(h < fhc.SumOp(terms[:-1]))
        self.assertTrue(fhc.SumOp([fhc.HoppingOp((0,), (3,), 0, 1.2)]) < h)
        self.assertTrue(h < fhc.SumOp([fhc.HoppingOp((0,), (3,), 0, 1.4)]))
        self.assertTrue(h + fhc.ZeroOp() == h)
        self.assertTrue(h - fhc.ZeroOp() == h)
        hmat = h.as_field_operator().as_matrix((L,))
        h_ref = 0
        for term in terms:
            h_ref += term.as_field_operator().as_matrix((L,))
        self.assertEqual(spla.norm(hmat - h_ref), 0)
        self.assertAlmostEqual(h.norm_bound(), np.linalg.norm(hmat.todense(), ord=2), delta=1e-12)
        ht = h.translate((-7,))
        self.assertEqual(ht.support(), [(-7, 0), (-6, 1), (-4, 0), (-4, 1)])
        # norm bound for sum of number operators; hopping term has disjoint support
        h = fhc.SumOp([fhc.NumberOp((3,), 0, -0.5), fhc.HoppingOp((0,), (1,), 0, -1.3), fhc.ProductOp([fhc.NumberOp((2,), 0, 0.6), fhc.NumberOp((3,), 1, -0.7)], 1.1), fhc.NumberOp((3,), 1, -0.2)])
        hmat = h.as_field_operator().as_matrix((L,))
        self.assertAlmostEqual(h.norm_bound(), np.linalg.norm(hmat.todense(), ord=2), delta=1e-12)

    def test_quadratic_norm(self):
        """
        Test norm computation for an operator consisting only of quadratic terms.
        """
        rng = np.random.default_rng()
        La = 5
        Lb = 4
        # create two quadratic operators with disjoint support
        supp_a = range(La)
        supp_b = range(La, La + Lb)
        for sgn in [1, -1]:
            if sgn == 1:
                ha = fhc.SumOp(
                    [fhc.HoppingOp((i,), (j,), s, rng.standard_normal())
                         for i in supp_a for j in supp_a for s in [0, 1] if i != j] +
                    [fhc.NumberOp((i,), s, rng.standard_normal())
                         for i in supp_a for s in [0, 1]])
                hb = fhc.SumOp(
                    [fhc.HoppingOp((i,), (j,), s, rng.standard_normal())
                         for i in supp_b for j in supp_b for s in [0, 1] if i != j] +
                    [fhc.NumberOp((i,), s, rng.standard_normal())
                         for i in supp_b for s in [0, 1]])
            else:
                ha = fhc.SumOp(
                    [fhc.AntisymmHoppingOp((i,), (j,), s, rng.standard_normal())
                         for i in supp_a for j in supp_a for s in [0, 1] if i != j])
                hb = fhc.SumOp(
                    [fhc.AntisymmHoppingOp((i,), (j,), s, rng.standard_normal())
                         for i in supp_b for j in supp_b for s in [0, 1] if i != j])
            cmt_a = ha.as_field_operator().as_compact_matrix().todense()
            cmt_b = hb.as_field_operator().as_compact_matrix().todense()
            self.assertTrue(np.allclose(cmt_a, sgn * cmt_a.T))
            self.assertTrue(np.allclose(cmt_b, sgn * cmt_b.T))
            eig_a = np.linalg.eigvalsh((1 if sgn == 1 else 1j) * cmt_a)
            eig_b = np.linalg.eigvalsh((1 if sgn == 1 else 1j) * cmt_b)
            # construction only works for unitarily diagonalizable operators
            nrm_ref = max(np.abs(np.add.outer(eig_a, eig_b).reshape(-1)))
            self.assertAlmostEqual((ha + hb).norm_bound(), nrm_ref, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
