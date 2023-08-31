from numbers import Rational
from fractions import Fraction
from warnings import warn
import numpy as np
from fh_comm.hamiltonian_ops import HamiltonianOp, ZeroOp, ProductOp, SumOp, NumberOp
from fh_comm.commutator import commutator
from fh_comm.lattice import SubLattice


def simplify(h: HamiltonianOp) -> HamiltonianOp:
    """
    Simplify a Hamiltonian operator expression.
    """
    return _factorize_numops(_expand_simplify(h))


def _expand_simplify(h: HamiltonianOp) -> HamiltonianOp:
    """
    Expand and simplify a Hamiltonian operator expression.
    """
    if isinstance(h, ProductOp):
        # expand a product of sums into a sum of products
        # first occurrence of a SumOp
        contains_sum = False
        # pylint: disable=consider-using-enumerate
        for i in range(len(h.ops)):
            if isinstance(h.ops[i], SumOp):
                contains_sum = True
                break
        if contains_sum:
            h = _expand_simplify(SumOp([ProductOp(h.ops[:i] + [term] + h.ops[i+1:], h.coeff) for term in h.ops[i].terms]))
        elif len(h.ops) == 1:
            # substitute simplified single operator for product
            h = _expand_simplify(h.coeff * h.ops[0])
        else:
            # recursively simplify operators in product
            h = ProductOp([_expand_simplify(op) for op in h.ops], h.coeff)
            # rearrange operators via bubble-sort (have to check pairwise commuting property)
            changed = True
            while changed:
                changed = False
                for i in range(len(h.ops) - 1):
                    if h.ops[i+1] < h.ops[i]:
                        if commutator(h.ops[i], h.ops[i+1]).is_zero():
                            tmp = h.ops[i]
                            h.ops[i] = h.ops[i+1]
                            h.ops[i+1] = tmp
                            changed = True
    elif isinstance(h, SumOp):
        h = SumOp(sorted([_expand_simplify(term) for term in h.terms]))
        if len(h.terms) == 1:
            # substitute simplified single operator for sum
            h = h.terms[0]
        else:
            # accumulate terms with same signature
            # pylint: disable=consider-using-enumerate
            for i in range(len(h.terms)):
                if isinstance(h.terms[i], ZeroOp):
                    continue
                for j in range(i+1, len(h.terms)):
                    if h.terms[i].proportional(h.terms[j]):
                        h.terms[i] += h.terms[j]
                        # assign zero operator to preserve indices; will be filtered out below
                        h.terms[j] = ZeroOp()
            h = SumOp(h.terms)
    return h


def _factorize_numops(h: HamiltonianOp) -> HamiltonianOp:
    """
    Factor out number operators in a Hamiltonian operator expression, if possible.
    """
    if not isinstance(h, SumOp):
        return h
    changed = True
    while changed:
        changed = False
        # pylint: disable=consider-using-enumerate
        for i in range(len(h.terms)):
            if not isinstance(h.terms[i], ProductOp):
                continue
            prod1 = h.terms[i]
            if len(prod1.ops) <= 1:
                continue
            # do not attempt to factor individual number operators,
            # since this would miss quadratic or higher-order factorizations
            # like n_i^2 - 2 n_i n_j + n_j^2 = (n_i - n_j)^2
            if prod1.is_numop_product():
                continue
            for j in range(i+1, len(h.terms)):
                if not isinstance(h.terms[j], ProductOp):
                    continue
                prod2 = h.terms[j]
                if len(prod2.ops) <= 1:
                    continue
                if prod2.is_numop_product():
                    continue
                # try to match the pattern
                # prod1 = a b ... d ni ... nk e ... f
                # prod2 = a b ... d nj ... nl e ... f
                # where a, ..., f are not number operators,
                # and ni, ..., nl can also be sums or products of number operators
                subprod = True
                for ks in range(min(len(prod1.ops), len(prod2.ops))):
                    if prod1.ops[ks] != prod2.ops[ks]:
                        subprod = False
                        break
                if subprod:
                    # one product operator is contained in the other; do not factorize
                    continue
                subprod = True
                for ke in range(min(len(prod1.ops), len(prod2.ops))):
                    if prod1.ops[-1-ke] != prod2.ops[-1-ke]:
                        subprod = False
                        break
                if subprod:
                    # one product operator is contained in the other; do not factorize
                    continue
                if not all(_consists_of_number_ops(prod1.ops[k]) for k in range(ks, len(prod1.ops) - ke)):
                    continue
                if not all(_consists_of_number_ops(prod2.ops[k]) for k in range(ks, len(prod2.ops) - ke)):
                    continue
                # include neighboring number operators
                while ks > 0 and _consists_of_number_ops(prod1.ops[ks - 1]):
                    assert _consists_of_number_ops(prod2.ops[ks - 1])
                    ks -= 1
                while ke > 0 and _consists_of_number_ops(prod1.ops[-ke]):
                    assert _consists_of_number_ops(prod2.ops[-ke])
                    ke -= 1
                # relative scaling factor
                if isinstance(prod1.coeff, Rational) and isinstance(prod2.coeff, Rational):
                    c = Fraction(prod2.coeff, prod1.coeff)
                    if c.denominator == 1:
                        c = c.numerator
                else:
                    c = prod2.coeff / prod1.coeff
                inner_sum = _expand_simplify(SumOp([ProductOp(prod1.ops[ks:len(prod1.ops)-ke], 1),
                                                    ProductOp(prod2.ops[ks:len(prod2.ops)-ke], c)]))
                assert inner_sum.is_numop_sum()
                prod1 = ProductOp(prod1.ops[:ks] + [inner_sum] + prod1.ops[len(prod1.ops)-ke:], prod1.coeff)
                h.terms[i] = prod1
                # assign zero operator to preserve indices; will be filtered out below
                h.terms[j] = ZeroOp()
                changed = True
        h = SumOp(h.terms)
    return h


def _consists_of_number_ops(h: HamiltonianOp) -> bool:
    """
    Whether a Hamiltonian operator consists only of number operators
    or sums and products of them.
    """
    return (isinstance(h, NumberOp) or
           (isinstance(h, ProductOp) and h.is_numop_product()) or
           (isinstance(h, SumOp) and h.is_numop_sum()))


def translate_origin(h: HamiltonianOp, translatt: SubLattice, bias: float = 0) -> HamiltonianOp:
    """
    Translate operators such that their support is centered around the origin,
    assuming translation invariance with respect to `translatt`.
    """
    if isinstance(h, SumOp):
        # translate terms individually
        h = SumOp([translate_origin(term, translatt, bias) for term in h.terms])
        # try to condense number operator terms further
        nidx = []
        for i, term in enumerate(h.terms):
            if isinstance(term, NumberOp) or (isinstance(term, ProductOp) and term.is_numop_product()):
                nidx.append(i)
        if len(nidx) > 1:
            maxiter = 4 * len(nidx)
            for _ in range(maxiter):
                changed = False
                # support center points
                scpt = []
                for i in nidx:
                    # support on lattice (without spin)
                    supp = list(set([s[:-1] for s in h.terms[i].support()]))
                    scpt.append(np.mean(np.array(supp), axis=0))
                # most distant center
                k = np.argmax([np.linalg.norm(pt) for pt in scpt])
                # center of other points
                ctr = sum(scpt[i] for i in range(len(scpt)) if i != k) / (len(scpt) - 1)
                shift = translatt.nearest_point(scpt[k] - ctr)
                if any(shift):
                    h.terms[nidx[k]] = h.terms[nidx[k]].translate([-x for x in shift])
                    changed = True
                if not changed:
                    break
            else:
                warn(f"maxiter = {maxiter} reached in 'translate_origin'")
        return h

    # support on lattice (without spin)
    supp = list(set([s[:-1] for s in h.support()]))
    # check for empty support, e.g., ZeroOp
    if not supp:
        return h
    shift = translatt.nearest_point(np.mean(np.array(supp), axis=0) - bias)
    return h.translate([-x for x in shift])
