import unittest
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import fh_comm as fhc


class TestFieldOps(unittest.TestCase):

    def test_ops(self):
        """
        Test field operator functionality.
        """
        # construct a sub-lattice
        sublatt = fhc.SubLattice([[2]])

        # a^{\dagger}_{i\sigma} a_{i\sigma} for \sigma = 1
        ad_a_dn = fhc.FieldOp([
            fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_CREATE,  (3,), 1),
                                fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_ANNIHIL, (3,), 1)], 0.7)])

        # number operator n_{i\sigma} for \sigma = 1
        num_dn = fhc.FieldOp([
            fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_NUMBER,  (3,), 1)], 0.7)])
        # compare
        self.assertEqual(spla.norm(ad_a_dn.as_matrix((5,)) - num_dn.as_matrix((5,))), 0)
        self.assertEqual(spla.norm(ad_a_dn.as_matrix((5,), sublatt) - num_dn.as_matrix((5,), sublatt)), 0)

        # a_{i\sigma} a^{\dagger}_{i\sigma} for \sigma = 0
        a_ad_up = fhc.FieldOp([
            fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_ANNIHIL, (2,), 0),
                                fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_CREATE,  (2,), 0)], 1.2)])
        # commutator should be zero since operators act on different sites
        self.assertEqual(spla.norm(comm(ad_a_dn, a_ad_up).as_matrix((5,), sublatt)), 0)

        # operators are shifted by a sub-lattice vector
        ad_up = fhc.FieldOp([fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_CREATE,  (3,), 0)], 0.25)])
        a_up  = fhc.FieldOp([fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_ANNIHIL, (1,), 0)], 4.0)])
        # fermionic anti-commutation relations
        self.assertEqual(spla.norm(
            anti_comm(ad_up.as_matrix((2,)), a_up.as_matrix((2,)))
            - sparse.identity(2**4)), 0)
        # factor 2 since there are 2 instances on sub-lattice
        self.assertEqual(spla.norm(
            anti_comm(ad_up.as_matrix((4,), sublatt), a_up.as_matrix((4,), sublatt))
            - 2*sparse.identity(2**8)), 0)

    def test_comm_n_a(self):
        """
        Verify that [n, a†] == a† and [a, n] == a.
        """
        for nmodes in range(1, 8):
            clist, alist, nlist = fhc.construct_fermionic_operators(nmodes)
            for i in range(nmodes):
                c = clist[i]
                a = alist[i]
                n = nlist[i]
                self.assertEqual(spla.norm(comm(n, c) - c), 0)
                self.assertEqual(spla.norm(comm(a, n) - a), 0)
        # use a numerical "orbital"
        rng = np.random.default_rng()
        x = crandn(5, rng)
        x /= np.linalg.norm(x)
        a = orbital_annihil_op(x)
        n = a.conj().T @ a
        self.assertAlmostEqual(spla.norm(comm(n, a.conj().T) - a.conj().T), 0)
        self.assertAlmostEqual(spla.norm(comm(a, n) - a), 0)

    def test_comm_n_hop(self):
        """
        Verify that [ni, ai† aj + aj† ai] == ai† aj - aj† ai.
        """
        for nmodes in range(1, 8):
            _, alist, nlist = fhc.construct_fermionic_operators(nmodes)
            for i in range(nmodes):
                for j in range(nmodes):
                    if i == j:
                        continue
                    ni = nlist[i]
                    ai = alist[i]
                    aj = alist[j]
                    self.assertEqual(spla.norm(
                        comm(ni, ai.T @ aj + aj.T @ ai)
                          - (ai.T @ aj - aj.T @ ai)), 0)

    def test_comm_hop_hop(self):
        """
        Verify that [ai† aj + aj† ai, ai† ak + ak† ai] == aj† ak - ak† aj.
        """
        for nmodes in range(1, 8):
            _, alist, _ = fhc.construct_fermionic_operators(nmodes)
            for i in range(nmodes):
                for j in range(nmodes):
                    for k in range(nmodes):
                        if i == j or j == k or k == i:
                            continue
                        ai = alist[i]
                        aj = alist[j]
                        ak = alist[k]
                        self.assertEqual(spla.norm(
                            comm(ai.T @ aj + aj.T @ ai, ai.T @ ak + ak.T @ ai)
                              - (aj.T @ ak - ak.T @ aj)), 0)

    def test_as_compact_matrix(self):
        """
        Verify that spectral norm of "compact" matrix representation matches norm for full lattice.
        """
        latt_shape = (2, 2)
        op = fhc.FieldOp([
            fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_ANNIHIL, (1, 0), 1),
                                fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_CREATE,  (1, 1), 0)], 1.1),
            fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_NUMBER,  (0, 1), 1)], -0.7),
            fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_CREATE,  (0, 1), 1),
                                fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_ANNIHIL, (1, 1), 0),
                                fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_ANNIHIL, (0, 1), 0)], 0.3)])
        op_cmt = op.as_compact_matrix()
        op_mat = op.as_matrix(latt_shape)
        self.assertAlmostEqual(np.linalg.norm(op_cmt.todense(), ord=2),
                               np.linalg.norm(op_mat.todense(), ord=2), delta=1e-12)

    def test_quadratic_coefficients(self):
        """
        Test extraction of quadratic coefficients.
        """
        # general quadratic field operator
        op = fhc.FieldOp([
            fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_CREATE,  (3,), 0),
                                fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_ANNIHIL, (2,), 1)], 1.3),
            fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_NUMBER,  (2,), 1)], -0.7),
            fhc.ProductFieldOp([fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_CREATE,  (1,), 0),
                                fhc.ElementaryFieldOp(fhc.FieldOpType.FERMI_ANNIHIL, (2,), 1)], 0.5)])
        op_cmt = op.as_compact_matrix()
        # construct matrix based on quadratic coefficients
        h = op.quadratic_coefficients()
        L = len(op.support())
        clist, alist, _ = fhc.construct_fermionic_operators(L)
        cmt_ref = sparse.csr_matrix((2**L, 2**L))
        for i in range(L):
            for j in range(L):
                cmt_ref += h[i, j] * (clist[i] @ alist[j])
        # compare
        self.assertAlmostEqual(spla.norm(op_cmt - cmt_ref), 0, delta=1e-12)


def anti_comm(a, b):
    """
    Anti-commutator {a, b} = a b + b a.
    """
    return a @ b + b @ a


def comm(a, b):
    """
    Commutator [a, b] = a b - b a.
    """
    return a @ b - b @ a


def orbital_create_op(x):
    """
    Fermionic "orbital" creation operator.
    """
    x = np.array(x, copy=False)
    nmodes = len(x)
    clist, _, _ = fhc.construct_fermionic_operators(nmodes)
    return sum(x[i] * clist[i] for i in range(nmodes))


def orbital_annihil_op(x):
    """
    Fermionic "orbital" annihilation operator.
    """
    # anti-linear with respect to coefficients in `x`
    return orbital_create_op(x).conj().T


def crandn(size=None, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None: rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


if __name__ == "__main__":
    unittest.main()
