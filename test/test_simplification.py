from fractions import Fraction
import unittest
import scipy.sparse.linalg as spla
import fh_comm as fhc


class TestSimplification(unittest.TestCase):

    def test_expand(self):
        """
        Test term expansion.
        """
        prod = fhc.ProductOp([
            fhc.SumOp([fhc.AntisymmHoppingOp((1,), (3,), 1, 0.2),
                        fhc.HoppingOp((0,), (3,), 0, 1.3)]),
            fhc.NumberOp((3,), 0, 0.5),
            fhc.SumOp([fhc.NumberOp((2,), 1, 1.2),
                        fhc.ProductOp([fhc.HoppingOp((1,), (2,), 0, 1.3),
                                      fhc.AntisymmHoppingOp((1,), (3,), 1, 0.2)], 1.1)])], 0.7)
        self.assertEqual(prod.fermi_weight, 8)
        psmp = fhc.simplify(prod)
        self.assertTrue(isinstance(psmp, fhc.SumOp))
        self.assertAlmostEqual(
            spla.norm(prod.as_field_operator().as_matrix((4,))
                    - psmp.as_field_operator().as_matrix((4,))), 0)
        self.assertEqual(psmp.fermi_weight, 8)

    def test_accumulate(self):
        """
        Test accumulation of terms with same signature.
        """
        h = fhc.SumOp([fhc.ProductOp([fhc.NumberOp((2,), 0, 0.4),
                                      fhc.HoppingOp((1,), (3,), 0, 0.7)], 0.9),
                        fhc.AntisymmHoppingOp((1,), (2,), 1, 0.2),
                        fhc.ProductOp([fhc.HoppingOp((1,), (3,), 0, 1.3),
                                      fhc.NumberOp((2,), 0, 0.5)], 1.1)])
        self.assertEqual(len(h.terms), 3)
        hs = fhc.simplify(h)
        self.assertEqual(len(hs.terms), 2)
        self.assertAlmostEqual(
            spla.norm(h.as_field_operator().as_matrix((4,))
                   - hs.as_field_operator().as_matrix((4,))), 0)

    def test_factorize_numops(self):
        """
        Test number operator factorization.
        """
        # number operators differ
        prod1 = fhc.ProductOp([fhc.AntisymmHoppingOp((1,), (2,), 1, Fraction(13, 10)),
                               fhc.NumberOp((2,), 1, Fraction(2, 5)),
                               fhc.HoppingOp((0,), (2,), 0, Fraction( 3, -7)),
                               fhc.NumberOp((1,), 1, Fraction(3, 10))], Fraction(-6, 11))
        prod2 = fhc.ProductOp([fhc.AntisymmHoppingOp((1,), (2,), 1, Fraction(1, 5)),
                               fhc.NumberOp((1,), 1, Fraction(9, -17)),
                               fhc.HoppingOp((0,), (2,), 0, Fraction(-9, 13))], Fraction(13, 3))
        prod3 = fhc.ProductOp([fhc.AntisymmHoppingOp((1,), (2,), 1, Fraction(-7, 17)),
                               fhc.NumberOp((3,), 0, Fraction(2, 5)),
                               fhc.HoppingOp((0,), (2,), 0, Fraction(-1, 6))], Fraction(-5, 6))
        h = fhc.SumOp([prod1, fhc.HoppingOp((0,), (1,), 0, Fraction(6, 5)), prod2, prod3])
        hs = fhc.simplify(h)
        self.assertEqual(len(hs.terms), 2)
        self.assertTrue(isinstance(hs.terms[1], fhc.ProductOp))
        self.assertTrue(any(isinstance(op, fhc.SumOp) for op in hs.terms[1].ops))
        self.assertAlmostEqual(
            spla.norm(h.as_field_operator().as_matrix((4,))
                   - hs.as_field_operator().as_matrix((4,))), 0)

    def test_translate(self):
        """
        Test translation with respect to sublattice.
        """
        translatt = fhc.SubLattice([[5, 2], [3, -1]])
        h = fhc.SumOp([fhc.ProductOp([fhc.NumberOp((2, 1), 0, 0.4),
                                      fhc.HoppingOp((2, 0), ( 1, -1), 1, 0.7)], 0.9),
                       fhc.NumberOp((-2, 7), 0, 0.3),
                       fhc.AntisymmHoppingOp((1, -1), (2, 5), 1, 0.2),
                       fhc.ProductOp([fhc.HoppingOp((0, 1), (-1, 0), 1, 1.3),
                                      fhc.NumberOp(( 0, -3), 0, 0.5)], 1.1),
                       fhc.ProductOp([fhc.NumberOp(( 2, -2), 0, 1.3),
                                      fhc.NumberOp((-4, -7), 1, 0.6)], -1.8),
                       fhc.NumberOp(( 3, 2), 1, 0.4)])
        hs = fhc.simplify(fhc.translate_origin(h, translatt))
        # the two product operators can be grouped together after translation
        self.assertEqual(len(hs.terms), len(h.terms) - 1)
        diff = fhc.simplify(fhc.translate_origin(fhc.simplify(h - hs), translatt))
        self.assertTrue(diff.is_zero())


if __name__ == "__main__":
    unittest.main()
