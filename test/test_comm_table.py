import itertools
import unittest
import scipy.sparse.linalg as spla
import fh_comm as fhc


class TestCommTable(unittest.TestCase):

    def test(self):
        # lattice size
        L = 4
        # construct a sub-lattice for translations
        translatt = fhc.SubLattice([[2]])
        # operators
        hlist = [fhc.SumOp([fhc.HoppingOp((0,), (2,), 0, 1.3), fhc.HoppingOp((0,), (2,), 1, 0.8)]),
                 fhc.ProductOp([fhc.NumberOp((2,), 0, 0.5), fhc.AntisymmHoppingOp((1,), (3,), 1, 0.2)], 0.6),
                 fhc.ProductOp([fhc.HoppingOp((2,), (1,), 1, 0.4), fhc.SumOp([fhc.AntisymmHoppingOp((3,), (1,), 1, 0.7), fhc.NumberOp((1,), 0, -0.3)])], 1.3)]
        # evaluate nested commutators between operators up to depth 3
        comm_tab = fhc.NestedCommutatorTable(hlist, 3)
        comt_tab = fhc.NestedCommutatorTable(hlist, 3, translatt)
        # matrix representations of operators
        hlist_mat = [h.as_field_operator().as_matrix((L,)) for h in hlist]
        htlst_mat = [h.as_field_operator().as_matrix((L,), translatt) for h in hlist]
        for commidx in itertools.product(range(len(hlist)), repeat=3):
            c_ref = comm(hlist_mat[commidx[2]], comm(hlist_mat[commidx[1]], hlist_mat[commidx[0]]))
            ctref = comm(htlst_mat[commidx[2]], comm(htlst_mat[commidx[1]], htlst_mat[commidx[0]]))
            self.assertAlmostEqual(spla.norm(comm_tab.table(2)[commidx[0]][commidx[1]][commidx[2]].as_field_operator().as_matrix((L,)) - c_ref), 0)
            self.assertAlmostEqual(spla.norm(comt_tab.table(2)[commidx[0]][commidx[1]][commidx[2]].as_field_operator().as_matrix((L,), translatt) - ctref), 0)


def comm(a, b):
    """
    Commutator [a, b] = a b - b a.
    """
    return a @ b - b @ a


if __name__ == "__main__":
    unittest.main()
