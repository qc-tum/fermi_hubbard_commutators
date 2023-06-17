import numpy as np


class SplittingMethod:
    """
    Splitting method described by the number of (Hamiltonian) terms
    (typically two, as for even-odd splitting), indices into these terms,
    and corresponding coefficients (time sub-step coefficients).
    """
    def __init__(self, nterms: int, indices, coeffs, order: int):
        # consistency check
        for i in indices:
            if i < 0 or i >= nterms:
                raise ValueError(f"index {i} out of range, must be between 0 and {nterms}")
        if len(coeffs) != len(indices):
            raise ValueError("length of coefficient list and index list must agree")
        weights = np.zeros(nterms)
        for i, c in zip(indices, coeffs):
            weights[i] += c
        if not np.allclose(weights, np.ones(nterms)):
            raise ValueError("weights for each term do not sum to 1")
        self.nterms  = nterms
        self.indices = list(indices)
        self.coeffs  = list(coeffs)
        self.order   = order

    @classmethod
    def trotter(cls, nterms: int):
        """
        Construct the first-order Lie-Trotter splitting formula.
        """
        return cls(nterms, range(nterms), nterms * [1], 1)

    @classmethod
    def suzuki(cls, nterms: int, k: int):
        """
        Construct the Suzuki product rule for order `2 k`.
        """
        indices, coeffs = _construct_suzuki_indices_coeffs(nterms, k)
        return cls(nterms, indices, coeffs, 2*k)

    @classmethod
    def yoshida4(cls, nterms: int = 2):
        """
        Symmetric integration method by Yoshida of order 4.

        Reference:
            Haruo Yoshida
            Construction of higher order symplectic integrators
            Phys. Lett. A 150, 262-268 (1990)
        """
        c1 = 0.5/(2 - 2**(1/3))
        c2 = 0.5*(1 - 1/(2 - 2**(1/3)))
        d1 = 2*c1
        d2 = 1 - 1/(1 - 1/2**(2/3))
        if nterms == 2:
            coeffs  = [c1, d1, c2, d2, c2, d1, c1]
            indices = [0,  1,  0,  1,  0,  1,  0 ]
        elif nterms == 3:
            e2 = 0.5*d2
            coeffs  = [c1, c1, d1, c1, c2, e2, d2, e2, c2, c1, d1, c1, c1]
            indices = [0,  1,  2,  1,  0,  1,  2,  1,  0,  1,  2,  1,  0 ]
        else:
            raise ValueError(f"integration method by Yoshida for {nterms} not supported")
        return cls(nterms, indices, coeffs, 4)

    @classmethod
    def mclachlan4(cls, m: int):
        """
        RKN method of order 4 by Robert I. McLachlan.

        Reference:
            Robert I. McLachlan
            On the numerical integration of ordinary differential equations by symmetric composition methods
            SIAM J. Sci. Comput. 16, 151-168 (1995)
        """
        if m == 4:
            a1 =     (642 + np.sqrt(471)) / 3924
            a2 =  121*(12 - np.sqrt(471)) / 3924
            a3 =  1 - 2*(a1 + a2)
            b1 =  6/11
            b2 =  0.5 - b1
            coeffs  = [a1, b1, a2, b2, a3, b2, a2, b1, a1]
            indices = [0,  1,  0,  1,  0,  1,  0,  1,  0 ]
            return cls(2, indices, coeffs, 4)
        if m == 5:
            a1 =  (14 -   np.sqrt(19)) / 108
            a2 =  (20 - 7*np.sqrt(19)) / 108
            a3 =  0.5 - (a1 + a2)
            b1 =  2/5
            b2 = -1/10
            b3 = 1 - 2*(b1 + b2)
            coeffs  = [a1, b1, a2, b2, a3, b3, a3, b2, a2, b1, a1]
            indices = [0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0 ]
            return cls(2, indices, coeffs, 4)
        raise ValueError(f"only m = 4 or m = 5 supported, received m = {m}")

    @classmethod
    def ak_11_4(cls):
        """
        Symmetric AK 11-4 method by Auzinger et al. of order 4 for three terms.

        Reference:
            Winfried Auzinger, Harald Hofstätter, David Ketcheson, Othmar Koch
            Practical splitting methods for the adaptive integration of nonlinear evolution equations.
            Part I: Construction of optimized schemes and pairs of schemes
            BIT Numer. Math. 57, 55-74 (2017)
        """
        a1 =  0.257069044488538534
        b1 =  0.296061717549380091
        c1 =  0.592448417648034871
        a2 =  0.432582164538475621
        b2 =  0.704720077493718759
        c2 =  0.819259857623654322
        a3 = -0.031637836548173035
        b3 = -0.046163676369010239
        c3 = -0.911708275271689193
        a4 = -0.158013372478841120
        b4 = -0.909236237348177222
        coeffs  = [a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, a4, c3, b3, a3, c2, b2, a2, c1, b1, a1]
        indices = [0,  1,  2,  0,  1,  2,  0,  1,  2,  0,  1,  0,  2,  1,  0,  2,  1,  0,  2,  1,  0 ]
        return cls(3, indices, coeffs, 4)

    @property
    def num_terms(self):
        """
        Number of (Hamiltonian) terms.
        """
        return self.nterms

    @property
    def num_layers(self):
        """
        Number of layers (substeps).
        """
        return len(self.coeffs)

    def __str__(self):
        """
        String representation of the product rule.
        """
        return f"Splitting method of order {self.order} for {self.nterms} terms using {self.num_layers} layers,\n  indices: {self.indices}\n  coeffs:  {self.coeffs}"


def _construct_suzuki_indices_coeffs(nterms: int, k: int):
    """
    Recursively construct the Suzuki product rule indices and coefficients for order `2 k`.
    """
    if k <= 0:
        raise ValueError(f"`k` must be a positive integer, received {k}")
    if k == 1:
        indices = list(range(nterms)) + list(reversed(range(nterms)))
        coeffs = (2*nterms) * (0.5,)
    else:
        uk = 1./(4 - 4**(1./(2*k-1)))
        ik1, ck1 = _construct_suzuki_indices_coeffs(nterms, k - 1)
        ck1_uk = [uk*c for c in ck1]
        ck1_14uk = [(1 - 4*uk)*c for c in ck1]
        indices = ik1 + ik1 + ik1 + ik1 + ik1
        coeffs = ck1_uk + ck1_uk + ck1_14uk + ck1_uk + ck1_uk
    return merge_layers(indices, coeffs)


def merge_layers(indices, coeffs):
    """
    Merge neighboring layers with the same index.
    """
    assert len(coeffs) == len(indices)
    mindices = [indices[0]]
    mcoeffs  = [coeffs[0]]
    for i, c in zip(indices[1:], coeffs[1:]):
        if mindices[-1] == i:
            mcoeffs[-1] += c
        else:
            mindices.append(i)
            mcoeffs.append(c)
    return mindices, mcoeffs
