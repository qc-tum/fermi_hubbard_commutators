import itertools
import unittest
import fh_comm as fhc


class TestCombinatorics(unittest.TestCase):

    def test_multinomial(self):
        """
        Test multinomial evaluation.
        """
        self.assertEqual(fhc.multinomial((3, 2, 7)), 7920)

    def test_integer_sum_tuples(self):
        """
        Test integer sum tuple generator.
        """
        for s in [0, 4, 5]:
            for nbins in [1, 3, 4]:
                g = fhc.integer_sum_tuples(s, nbins)
                # reference implementation
                tupref = []
                for t in itertools.product(range(s + 1), repeat=nbins):
                    if not sum(t) == s:
                        continue
                    tupref.append(t)
                self.assertEqual(list(g), tupref)


if __name__ == "__main__":
    unittest.main()
