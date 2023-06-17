import unittest
import fh_comm as fhc


class TestSplittingMethod(unittest.TestCase):

    def test_trotter(self):
        """
        Consistency checks for the Lie-Trotter splitting method.
        """
        for nterms in range(2, 4):
            rule = fhc.SplittingMethod.trotter(nterms)
            # more refined consistency checks are performed within __init__
            self.assertAlmostEqual(sum(rule.coeffs), rule.nterms)

    def test_suzuki(self):
        """
        Consistency checks for the Suzuki splitting methods.
        """
        for nterms in range(2, 4):
            for k in range(1, 3):
                rule = fhc.SplittingMethod.suzuki(nterms, k)
                # more refined consistency checks are performed within __init__
                self.assertAlmostEqual(sum(rule.coeffs), rule.nterms)

    def test_yoshida(self):
        """
        Consistency checks for the splitting method by Yoshida of order 4.
        """
        for nterms in [2, 3]:
            rule = fhc.SplittingMethod.yoshida4(nterms)
            # more refined consistency checks are performed within __init__
            self.assertAlmostEqual(sum(rule.coeffs), rule.nterms)

    def test_mclachlan(self):
        """
        Consistency checks for splitting methods by Robert I. McLachlan.
        """
        for rule in [fhc.SplittingMethod.mclachlan4(m) for m in [4, 5]]:
            # more refined consistency checks are performed within __init__
            self.assertAlmostEqual(sum(rule.coeffs), rule.nterms)


if __name__ == "__main__":
    unittest.main()
