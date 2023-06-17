import math
from fractions import Fraction
import numpy as np
from .combinatorics import multinomial, integer_sum_tuples
from .splitting_method import SplittingMethod


class WeightedNestedCommutator:
    """
    Symbolic weighted nested commutator.

    A commutation index `i` is interpreted as
    [A_{i[-1]}, ...[A_{i[2]}, [A_{i[1]}, A_{i[0]}]]...]
    """
    def __init__(self, commidx, weight):
        self.commidx = commidx
        self.weight = weight

    def __str__(self):
        """
        String representation of the weighted nested commutator.
        """
        s = f"H_{self.commidx[0]}"
        for i in self.commidx[1:]:
            s = f"[H_{i}, {s}]"
        s = str(self.weight) + " * " + s
        return s


def commutator_bound(splitting: SplittingMethod, s: int):
    """
    Coefficients for commutator bounds on a splitting method (product rule).
    """
    weights = np.zeros((splitting.order + 1) * (splitting.num_terms,))
    for j in range(1, splitting.num_layers):
        bcoeff = np.zeros(splitting.num_terms)
        for i, c in zip(splitting.indices[:j], splitting.coeffs[:j]):
            bcoeff[i] += c
        for q in integer_sum_tuples(splitting.order, s-j if j < s else j-s+1):
            if q[0] == 0:
                continue
            mq = multinomial(q)
            for k in range(splitting.num_terms):
                if bcoeff[k] == 0:
                    continue
                commidx = (k,)
                w = bcoeff[k]
                for i in range(len(q)):
                    l = j + i if j < s else j - i
                    commidx += q[i] * (splitting.indices[l],)
                    w *= abs(splitting.coeffs[l])**q[i]
                assert len(commidx) == splitting.order + 1
                if commidx[0] == commidx[1]:
                    # [A, A] = 0
                    continue
                if commidx[0] > commidx[1]:
                    # [A, B] = -[B, A], and absolute values agree
                    commidx = (commidx[1], commidx[0]) + commidx[2:]
                weights[commidx] += mq * w
    weights /= math.factorial(splitting.order + 1)
    # assemble return value
    res = []
    for idx, w in np.ndenumerate(weights):
        if w != 0:
            res.append(WeightedNestedCommutator(idx, w))
    return res


def commutator_bound_strang(nterms: int):
    """
    Coefficients for commutator bound specifically for Strang (second-order Suzuki) formula.

    Reference: Proposition 10 in
        Andrew M. Childs, Yuan Su, Minh C. Tran, Nathan Wiebe, Shuchen Zhu
        Theory of Trotter error with commutator scaling
        Phys. Rev. X 11, 011020 (2021)
    """
    res = []
    for i in range(nterms):
        for j in range(i + 1, nterms):
            res.append(WeightedNestedCommutator((i, j, i), Fraction(1, 24)))
            for k in range(i + 1, nterms):
                res.append(WeightedNestedCommutator((i, j, k), Fraction(1, 12)))
    return res
