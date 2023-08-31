from collections.abc import Sequence
from fh_comm.lattice import SubLattice
from fh_comm.hamiltonian_ops import HamiltonianOp, HoppingOp, AntisymmHoppingOp, NumberOp, ZeroOp, ProductOp, SumOp


def commutator(a: HamiltonianOp, b: HamiltonianOp) -> HamiltonianOp:
    """
    Commutator between two fermionic Hamiltonian operator terms.
    """
    # manual pattern matching
    if isinstance(a, ZeroOp) or isinstance(b, ZeroOp):
        return ZeroOp()
    if isinstance(a, ProductOp):
        # commute `b` through the operators in `a`
        return SumOp([ProductOp(a.ops[:i] + [commutator(a.ops[i], b)] + a.ops[i+1:], a.coeff) for i in range(len(a.ops))])
    if isinstance(b, ProductOp):
        # commute `a` through the operators in `b`
        return SumOp([ProductOp(b.ops[:i] + [commutator(a, b.ops[i])] + b.ops[i+1:], b.coeff) for i in range(len(b.ops))])
    if isinstance(a, SumOp):
        return SumOp([commutator(term, b) for term in a.terms])
    if isinstance(b, SumOp):
        return SumOp([commutator(a, term) for term in b.terms])
    if isinstance(a, HoppingOp):
        if isinstance(b, HoppingOp):
            return _commutator_hopping(a, b)
        if isinstance(b, AntisymmHoppingOp):
            return _commutator_mixed_symm_hopping(a, b)
        if isinstance(b, NumberOp):
            return _commutator_hopping_number(a, b)
    elif isinstance(a, AntisymmHoppingOp):
        if isinstance(b, HoppingOp):
            # pylint: disable=arguments-out-of-order
            return -_commutator_mixed_symm_hopping(b, a)
        if isinstance(b, AntisymmHoppingOp):
            return _commutator_antisymm_hopping(a, b)
        if isinstance(b, NumberOp):
            return _commutator_antisymm_hopping_number(a, b)
    elif isinstance(a, NumberOp):
        if isinstance(b, HoppingOp):
            # pylint: disable=arguments-out-of-order
            return -_commutator_hopping_number(b, a)
        if isinstance(b, AntisymmHoppingOp):
            # pylint: disable=arguments-out-of-order
            return -_commutator_antisymm_hopping_number(b, a)
        if isinstance(b, NumberOp):
            return ZeroOp()
    # should never reach this point
    raise NotImplementedError()


def _commutator_hopping(a: HoppingOp, b: HoppingOp) -> HamiltonianOp:
    """
    Commutator between two fermionic hopping operators.
    """
    assert isinstance(a, HoppingOp) and isinstance(b, HoppingOp)
    if a.s != b.s:
        return ZeroOp()
    if a.i == b.i:
        if a.j == b.j:
            # operators are identical
            return ZeroOp()
        return AntisymmHoppingOp(a.j, b.j, a.s, a.coeff * b.coeff).standard_form()
    if a.i == b.j:
        # flip b.i <-> b.j
        return _commutator_hopping(a, b.flip())
    if a.j == b.i:
        # flip a.i <-> a.j
        return _commutator_hopping(a.flip(), b)
    if a.j == b.j:
        # flip a.i <-> a.j and b.i <-> b.j
        return _commutator_hopping(a.flip(), b.flip())
    # operators act on disjoint sites
    return ZeroOp()


def _commutator_mixed_symm_hopping(a: HoppingOp, b: AntisymmHoppingOp) -> HamiltonianOp:
    """
    Commutator between a symmetric and an anti-symmetric fermionic hopping operator.
    """
    assert isinstance(a, HoppingOp) and isinstance(b, AntisymmHoppingOp)
    if a.s != b.s:
        return ZeroOp()
    if b.i == b.j:
        # 'b' is the zero operator
        return ZeroOp()
    if a.i == b.i:
        if a.j == b.j:
            return SumOp([NumberOp(a.i, a.s, -2 * a.coeff * b.coeff), NumberOp(a.j, a.s, 2 * a.coeff * b.coeff)])
        return HoppingOp(a.j, b.j, a.s, a.coeff * b.coeff).standard_form()
    if a.i == b.j:
        # flip b.i <-> b.j
        return _commutator_mixed_symm_hopping(a, b.flip())
    if a.j == b.i:
        # flip a.i <-> a.j
        return _commutator_mixed_symm_hopping(a.flip(), b)
    if a.j == b.j:
        # flip a.i <-> a.j and b.i <-> b.j
        return _commutator_mixed_symm_hopping(a.flip(), b.flip())
    # operators act on disjoint sites
    return ZeroOp()


def _commutator_antisymm_hopping(a: AntisymmHoppingOp, b: AntisymmHoppingOp) -> HamiltonianOp:
    """
    Commutator between two anti-symmetric fermionic hopping operators.
    """
    assert isinstance(a, AntisymmHoppingOp) and isinstance(b, AntisymmHoppingOp)
    if a.s != b.s:
        return ZeroOp()
    if a.i == b.i:
        if a.j == b.j:
            # operators are identical
            return ZeroOp()
        return AntisymmHoppingOp(b.j, a.j, a.s, a.coeff * b.coeff).standard_form()
    if a.i == b.j:
        # flip b.i <-> b.j
        return _commutator_antisymm_hopping(a, b.flip())
    if a.j == b.i:
        # flip a.i <-> a.j
        return _commutator_antisymm_hopping(a.flip(), b)
    if a.j == b.j:
        # flip a.i <-> a.j and b.i <-> b.j
        return _commutator_antisymm_hopping(a.flip(), b.flip())
    # operators act on disjoint sites
    return ZeroOp()


def _commutator_hopping_number(a: HoppingOp, b: NumberOp) -> HamiltonianOp:
    """
    Commutator between a fermionic hopping and number operator.
    """
    assert isinstance(a, HoppingOp) and isinstance(b, NumberOp)
    if a.s != b.s:
        return ZeroOp()
    if a.i == b.i:
        return AntisymmHoppingOp(a.j, a.i, a.s, a.coeff * b.coeff).standard_form()
    if a.j == b.i:
        return AntisymmHoppingOp(a.i, a.j, a.s, a.coeff * b.coeff).standard_form()
    # operators act on disjoint sites
    return ZeroOp()


def _commutator_antisymm_hopping_number(a: AntisymmHoppingOp, b: NumberOp) -> HamiltonianOp:
    """
    Commutator between an anti-symmetric hopping and number operator.
    """
    assert isinstance(a, AntisymmHoppingOp) and isinstance(b, NumberOp)
    if a.s != b.s:
        return ZeroOp()
    if a.i == b.i:
        return HoppingOp(a.i, a.j, a.s, -a.coeff * b.coeff).standard_form()
    if a.j == b.i:
        return HoppingOp(a.i, a.j, a.s, a.coeff * b.coeff).standard_form()
    # operators act on disjoint sites
    return ZeroOp()


def commutator_translation(a: HamiltonianOp, b: HamiltonianOp, translatt: SubLattice) -> HamiltonianOp:
    """
    Commutator between two fermionic Hamiltonian operator terms,
    assuming that they consist of translated copies according to `translatt`.
    """
    if a.is_zero() or b.is_zero():
        # fast return to avoid issues with support region
        return ZeroOp()
    # generate shifted copies of b
    cmin_a, cmax_a = _support_region(a.support())
    cmin_b, cmax_b = _support_region(b.support())
    cmax = tuple(x - y + 1 for x, y in zip(cmax_a, cmin_b))
    cmin = tuple(x - y     for x, y in zip(cmin_a, cmax_b))
    ptlist = translatt.instantiate(cmin, cmax)
    ct = SumOp([])
    for pt in ptlist:
        ct += commutator(a, b.translate(pt))
    if ct.is_zero():
        return ZeroOp()
    return ct


def _support_region(supp: Sequence[tuple]):
    """
    Enclosing rectangular region enclosing the provided support points,
    ignoring the spin degree of freedom.
    """
    if not supp:
        return (), ()
    # support without spin
    cmin = list(supp[0][:-1])
    cmax = list(supp[0][:-1])
    for s in supp:
        for i in range(len(cmin)):
            if cmin[i] > s[i]:
                cmin[i] = s[i]
            elif cmax[i] < s[i]:
                cmax[i] = s[i]
    return tuple(cmin), tuple(cmax)
