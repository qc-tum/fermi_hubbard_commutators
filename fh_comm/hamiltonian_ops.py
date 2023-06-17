import abc
import math
from functools import total_ordering
from numbers import Rational
from collections.abc import Sequence
from warnings import warn
import numpy as np
from .field_ops import FieldOpType, ElementaryFieldOp, ProductFieldOp, FieldOp


@total_ordering
class HamiltonianOp(abc.ABC):
    """
    Parent class for fermionic Hamiltonian operators or commutators between them.
    """

    @abc.abstractmethod
    def __neg__(self):
        """
        Logical negation.
        """

    @abc.abstractmethod
    def __rmul__(self, other):
        """
        Logical scalar product.
        """

    @abc.abstractmethod
    def __add__(self, other):
        """
        Logical sum.
        """

    @abc.abstractmethod
    def __sub__(self, other):
        """
        Logical difference.
        """

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """
        Equality test.
        """

    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """

    @abc.abstractmethod
    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """

    @abc.abstractmethod
    def as_field_operator(self) -> FieldOp:
        """
        Represent the operator in terms of fermionic field operators.
        """

    @abc.abstractmethod
    def support(self) -> list[tuple]:
        """
        Support of the operator: lattice sites which it acts on (including spin).
        """

    @abc.abstractmethod
    def translate(self, shift: Sequence[int]):
        """
        Translate by `shift` and return the translated operator.
        """

    @abc.abstractmethod
    def normalize(self) -> tuple:
        """
        Return a normalized copy of the operator together with its scalar prefactor.
        """

    @abc.abstractmethod
    def is_zero(self) -> bool:
        """
        Indicate whether the operator acts as zero operator.
        """

    @property
    @abc.abstractmethod
    def fermi_weight(self) -> int:
        """
        Maximum number of fermionic creation and annihilation operators multiplied together.

        For example, the product of three number operators has weight 6.
        """

    @abc.abstractmethod
    def norm_bound(self) -> float:
        """
        Upper bound on the spectral norm of the operator.
        """

    # maximum number of modes defining matrix dimension
    # for which the exact spectral norm can be computed
    max_nmodes_exact_norm = 14


class HoppingOp(HamiltonianOp):
    r"""
    Hopping term :math:`a^{\dagger}_{i\sigma} a_{j\sigma} + h.c.`
    between sites `i` and `j` with spin `s`.
    """
    def __init__(self, i: Sequence[int], j: Sequence[int], s: int, coeff: float):
        self.i = tuple(i)
        self.j = tuple(j)
        assert self.i != self.j
        assert s in [0, 1]
        self.s = s
        self.coeff = coeff

    def __neg__(self):
        """
        Logical negation.
        """
        return HoppingOp(self.i, self.j, self.s, -self.coeff)

    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return HoppingOp(self.i, self.j, self.s, other * self.coeff)

    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if not self.proportional(other):
            raise ValueError("can only add another hopping term acting on same sites")
        return HoppingOp(self.i, self.j, self.s, self.coeff + other.coeff)

    def __sub__(self, other):
        """
        Logical difference.
        """
        return self + (-other)

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        c = "" if self.coeff == 1 else f"({self.coeff}) "
        return c + f"h_{{{self.i}, {self.j}, {'up' if self.s == 0 else 'dn'}}}"

    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, HoppingOp):
            if self.i == other.i and self.j == other.j and self.s == other.s and self.coeff == other.coeff:
                return True
        return False

    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, (AntisymmHoppingOp, NumberOp, ProductOp, SumOp)):
            return True
        if isinstance(other, ZeroOp):
            return False
        assert isinstance(other, HoppingOp)
        # lexicographical comparison
        return (self.s, self.i, self.j, self.coeff) < (other.s, other.i, other.j, other.coeff)

    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        if isinstance(other, HoppingOp):
            if self.i == other.i and self.j == other.j and self.s == other.s:
                return True
        return False

    def as_field_operator(self) -> FieldOp:
        """
        Represent the operator in terms of fermionic field operators.
        """
        return FieldOp([
            ProductFieldOp([ElementaryFieldOp(FieldOpType.FERMI_CREATE,  self.i, self.s),
                            ElementaryFieldOp(FieldOpType.FERMI_ANNIHIL, self.j, self.s)], self.coeff),
            ProductFieldOp([ElementaryFieldOp(FieldOpType.FERMI_CREATE,  self.j, self.s),
                            ElementaryFieldOp(FieldOpType.FERMI_ANNIHIL, self.i, self.s)], self.coeff)])

    def support(self) -> list[tuple]:
        """
        Support of the operator: lattice sites which it acts on (including spin).
        """
        return sorted([self.i + (self.s,), self.j + (self.s,)])

    def translate(self, shift: Sequence[int]):
        """
        Translate by `shift` and return the translated operator.
        """
        return HoppingOp(tuple(x + s for x, s in zip(self.i, shift)),
                         tuple(x + s for x, s in zip(self.j, shift)),
                         self.s, self.coeff)

    def normalize(self) -> tuple:
        """
        Return a normalized copy of the operator together with its scalar prefactor.
        """
        if self.coeff == 1:
            return self, 1
        return HoppingOp(self.i, self.j, self.s, 1), self.coeff

    def is_zero(self) -> bool:
        """
        Indicate whether the operator acts as zero operator.
        """
        return self.coeff == 0

    @property
    def fermi_weight(self) -> int:
        """
        Maximum number of fermionic creation and annihilation operators multiplied together.
        """
        return 2

    def norm_bound(self) -> float:
        """
        Upper bound on the spectral norm of the operator.
        """
        return abs(self.coeff)

    def flip(self):
        """
        Interchange i <-> j, resulting in logically the same operator.
        """
        return HoppingOp(self.j, self.i, self.s, self.coeff)

    def standard_form(self):
        """
        Return the "standard form" of the operator, such that i <= j (lexicographical ordering).
        """
        if self.i <= self.j:
            return self
        else:
            return self.flip()


class AntisymmHoppingOp(HamiltonianOp):
    r"""
    Anti-symmetric hopping term :math:`a^{\dagger}_{i\sigma} a_{j\sigma} - h.c.`
    between sites `i` and `j` with spin `s`.
    """
    def __init__(self, i: Sequence[int], j: Sequence[int], s: int, coeff: float):
        self.i = tuple(i)
        self.j = tuple(j)
        assert self.i != self.j
        assert s in [0, 1]
        self.s = s
        self.coeff = coeff

    def __neg__(self):
        """
        Logical negation.
        """
        return AntisymmHoppingOp(self.i, self.j, self.s, -self.coeff)

    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return AntisymmHoppingOp(self.i, self.j, self.s, other * self.coeff)

    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if not self.proportional(other):
            raise ValueError("can only add another anti-symmetric hopping term acting on same sites")
        return AntisymmHoppingOp(self.i, self.j, self.s, self.coeff + other.coeff)

    def __sub__(self, other):
        """
        Logical difference.
        """
        return self + (-other)

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        c = "" if self.coeff == 1 else f"({self.coeff}) "
        return c + f"g_{{{self.i}, {self.j}, {'up' if self.s == 0 else 'dn'}}}"

    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, AntisymmHoppingOp):
            if self.i == other.i and self.j == other.j and self.s == other.s and self.coeff == other.coeff:
                return True
        return False

    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, (NumberOp, ProductOp, SumOp)):
            return True
        if isinstance(other, (ZeroOp, HoppingOp)):
            return False
        assert isinstance(other, AntisymmHoppingOp)
        # lexicographical comparison
        return (self.s, self.i, self.j, self.coeff) < (other.s, other.i, other.j, other.coeff)

    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        if isinstance(other, AntisymmHoppingOp):
            if self.i == other.i and self.j == other.j and self.s == other.s:
                return True
        return False

    def as_field_operator(self) -> FieldOp:
        """
        Represent the operator in terms of fermionic field operators.
        """
        return FieldOp([
            ProductFieldOp([ElementaryFieldOp(FieldOpType.FERMI_CREATE,  self.i, self.s),
                            ElementaryFieldOp(FieldOpType.FERMI_ANNIHIL, self.j, self.s)],  self.coeff),
            ProductFieldOp([ElementaryFieldOp(FieldOpType.FERMI_CREATE,  self.j, self.s),
                            ElementaryFieldOp(FieldOpType.FERMI_ANNIHIL, self.i, self.s)], -self.coeff)])

    def support(self) -> list[tuple]:
        """
        Support of the operator: lattice sites which it acts on (including spin).
        """
        return sorted([self.i + (self.s,), self.j + (self.s,)])

    def translate(self, shift: Sequence[int]):
        """
        Translate by `shift` and return the translated operator.
        """
        return AntisymmHoppingOp(tuple(x + s for x, s in zip(self.i, shift)),
                                 tuple(x + s for x, s in zip(self.j, shift)),
                                 self.s, self.coeff)

    def normalize(self) -> tuple:
        """
        Return a normalized copy of the operator together with its scalar prefactor.
        """
        if self.coeff == 1:
            return self, 1
        return AntisymmHoppingOp(self.i, self.j, self.s, 1), self.coeff

    def is_zero(self) -> bool:
        """
        Indicate whether the operator acts as zero operator.
        """
        return self.coeff == 0

    @property
    def fermi_weight(self) -> int:
        """
        Maximum number of fermionic creation and annihilation operators multiplied together.
        """
        return 2

    def norm_bound(self) -> float:
        """
        Upper bound on the spectral norm of the operator.
        """
        return abs(self.coeff)

    def flip(self):
        """
        Interchange i <-> j and flip sign, resulting in logically the same operator.
        """
        return AntisymmHoppingOp(self.j, self.i, self.s, -self.coeff)

    def standard_form(self):
        """
        Return the "standard form" of the operator, such that i <= j (lexicographical ordering).
        """
        if self.i <= self.j:
            return self
        else:
            return self.flip()


class NumberOp(HamiltonianOp):
    r"""
    Number operator :math:`n_{i\sigma}`.
    """
    def __init__(self, i: Sequence[int], s: int, coeff: float):
        self.i = tuple(i)
        assert s in [0, 1]
        self.s = s
        self.coeff = coeff

    def __neg__(self):
        """
        Logical negation.
        """
        return NumberOp(self.i, self.s, -self.coeff)

    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return NumberOp(self.i, self.s, other * self.coeff)

    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if not self.proportional(other):
            raise ValueError("can only add another number operator acting on same site")
        return NumberOp(self.i, self.s, self.coeff + other.coeff)

    def __sub__(self, other):
        """
        Logical difference.
        """
        return self + (-other)

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        c = "" if self.coeff == 1 else f"({self.coeff}) "
        return c + f"n_{{{self.i}, {'up' if self.s == 0 else 'dn'}}}"

    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, NumberOp):
            if self.i == other.i and self.s == other.s and self.coeff == other.coeff:
                return True
        return False

    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, (ProductOp, SumOp)):
            return True
        if isinstance(other, (ZeroOp, HoppingOp, AntisymmHoppingOp)):
            return False
        assert isinstance(other, NumberOp)
        # lexicographical comparison
        return (self.s, self.i, self.coeff) < (other.s, other.i, other.coeff)

    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        if isinstance(other, NumberOp):
            if self.i == other.i and self.s == other.s:
                return True
        return False

    def as_field_operator(self) -> FieldOp:
        """
        Represent the operator in terms of fermionic field operators.
        """
        return FieldOp([
            ProductFieldOp([
                ElementaryFieldOp(FieldOpType.FERMI_NUMBER, self.i, self.s)], self.coeff)])

    def support(self) -> list[tuple]:
        """
        Support of the operator: lattice sites which it acts on (including spin).
        """
        return [self.i + (self.s,)]

    def translate(self, shift: Sequence[int]):
        """
        Translate by `shift` and return the translated operator.
        """
        return NumberOp(tuple(x + s for x, s in zip(self.i, shift)), self.s, self.coeff)

    def normalize(self) -> tuple:
        """
        Return a normalized copy of the operator together with its scalar prefactor.
        """
        if self.coeff == 1:
            return self, 1
        return NumberOp(self.i, self.s, 1), self.coeff

    def is_zero(self) -> bool:
        """
        Indicate whether the operator acts as zero operator.
        """
        return self.coeff == 0

    @property
    def fermi_weight(self) -> int:
        """
        Maximum number of fermionic creation and annihilation operators multiplied together.
        """
        return 2

    def norm_bound(self) -> float:
        """
        Upper bound on the spectral norm of the operator.
        """
        return abs(self.coeff)


class ZeroOp(HamiltonianOp):
    """
    Zero operator.
    """

    def __neg__(self):
        """
        Logical negation.
        """
        return ZeroOp()

    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return ZeroOp()

    def __add__(self, other):
        """
        Logical sum.
        """
        return other

    def __sub__(self, other):
        """
        Logical difference.
        """
        return -other

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        return "<zero op>"

    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, ZeroOp):
            return True
        return False

    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, ZeroOp):
            # ZeroOp equal to another ZeroOp
            return False
        return True

    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        if isinstance(other, ZeroOp):
            return True
        return False

    def as_field_operator(self) -> FieldOp:
        """
        Represent the operator in terms of fermionic field operators.
        """
        return FieldOp([])

    def support(self) -> list[tuple]:
        """
        Support of the operator: lattice sites which it acts on (including spin).
        """
        return []

    def translate(self, shift: Sequence[int]):
        """
        Translate by `shift` and return the translated operator.
        """
        return ZeroOp()

    def normalize(self) -> tuple:
        """
        Return a normalized copy of the operator together with its scalar prefactor.
        """
        return self, 1

    def is_zero(self) -> bool:
        """
        Indicate whether the operator acts as zero operator.
        """
        return True

    @property
    def fermi_weight(self) -> int:
        """
        Maximum number of fermionic creation and annihilation operators multiplied together.
        """
        return 0

    def norm_bound(self) -> float:
        """
        Upper bound on the spectral norm of the operator.
        """
        return 0


class ProductOp(HamiltonianOp):
    """
    Product of Hamiltonian operators.
    """
    def __init__(self, ops: Sequence[HamiltonianOp], coeff: float):
        self.ops = []
        self.coeff = coeff
        for op in ops:
            if op.is_zero():
                self.ops = [ZeroOp()]
                self.coeff = 1
                return
            # flatten nested products
            if isinstance(op, ProductOp):
                self.ops += op.ops
                self.coeff *= op.coeff
            else:
                opn, c = op.normalize()
                self.ops.append(opn)
                self.coeff *= c

    def __neg__(self):
        """
        Logical negation.
        """
        return ProductOp(self.ops, -self.coeff)

    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return ProductOp(self.ops, other * self.coeff)

    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if not self.proportional(other):
            raise ValueError("can only add another product operator with same factors")
        # assuming that each operator in product is normalized
        return ProductOp(self.ops, self.coeff + other.coeff)

    def __sub__(self, other):
        """
        Logical difference.
        """
        return self + (-other)

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        c = "" if self.coeff == 1 else f"({self.coeff}) "
        if not self.ops:
            # logical identity operator
            return c + "<empty product>"
        s = ""
        for op in self.ops:
            s += ("" if s == "" else " @ ") + "(" + str(op) + ")"
        return c + s

    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, ProductOp):
            if len(self.ops) == len(other.ops) and self.coeff == other.coeff:
                if all(op1 == op2 for op1, op2 in zip(self.ops, other.ops)):
                    return True
        return False

    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, SumOp):
            return True
        if isinstance(other, (ZeroOp, HoppingOp, AntisymmHoppingOp, NumberOp)):
            return False
        assert isinstance(other, ProductOp)
        if self.fermi_weight < other.fermi_weight:
            return True
        if other.fermi_weight < self.fermi_weight:
            return False
        # compare individual factors
        # note: zip only includes entries up to shorter length of both lists
        for op1, op2 in zip(self.ops, other.ops):
            if op1 < op2:
                return True
            if op2 < op1:
                return False
            assert op1 == op2
        # fermi weights and leading factors agree, so overall product operators must be proportional
        return self.coeff < other.coeff

    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        if isinstance(other, ProductOp):
            if len(self.ops) == len(other.ops):
                if all(op1.proportional(op2) for op1, op2 in zip(self.ops, other.ops)):
                    return True
        return False

    def as_field_operator(self) -> FieldOp:
        """
        Represent the operator in terms of fermionic field operators.
        """
        if not self.ops:
            raise RuntimeError("require at least one operator")
        fop = self.ops[0].as_field_operator()
        for op in self.ops[1:]:
            fop = fop @ op.as_field_operator()
        return self.coeff * fop

    def support(self) -> list[tuple]:
        """
        Support of the operator: lattice sites which it acts on (including spin).
        """
        s = []
        for op in self.ops:
            s += op.support()
        return sorted(list(set(s)))

    def translate(self, shift: Sequence[int]):
        """
        Translate by `shift` and return the translated operator.
        """
        return ProductOp([op.translate(shift) for op in self.ops], self.coeff)

    def normalize(self) -> tuple:
        """
        Return a normalized copy of the operator together with its scalar prefactor.
        """
        if self.coeff == 1:
            return self, 1
        return ProductOp(self.ops, 1), self.coeff

    def is_zero(self) -> bool:
        """
        Indicate whether the operator acts as zero operator.
        """
        if self.coeff == 0:
            return True
        for op in self.ops:
            if op.is_zero():
                return True
        # empty 'ops' logically corresponds to identity operation
        return False

    @property
    def fermi_weight(self) -> int:
        """
        Maximum number of fermionic creation and annihilation operators multiplied together.
        """
        return sum(op.fermi_weight for op in self.ops)

    def norm_bound(self) -> float:
        """
        Upper bound on the spectral norm of the operator.
        """
        if not self.ops:
            # logical identity operation
            return abs(self.coeff)
        nmodes = len(self.support())
        if nmodes <= HamiltonianOp.max_nmodes_exact_norm:
            cmt = self.as_field_operator().as_compact_matrix()
            return _spectral_norm_conserved_particles(nmodes, cmt)
        # resort to upper bound
        # use sub-multiplicative property of the spectral norm
        return abs(self.coeff) * math.prod([op.norm_bound() for op in self.ops])

    def is_numop_product(self) -> bool:
        """
        Whether the operator is a product of number operators.
        """
        return bool(self.ops) and all(isinstance(op, NumberOp) for op in self.ops)


class SumOp(HamiltonianOp):
    """
    Sum of Hamiltonian operators.
    """
    def __init__(self, terms: Sequence[HamiltonianOp]):
        self.terms = []
        for term in terms:
            # filter out zero operators
            if term.is_zero():
                continue
            # flatten nested sums
            if isinstance(term, SumOp):
                self.terms += term.terms
            else:
                self.terms.append(term)
        self.terms = sorted(self.terms)

    def __neg__(self):
        """
        Logical negation.
        """
        return SumOp([-term for term in self.terms])

    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return SumOp([other * term for term in self.terms])

    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if isinstance(other, SumOp):
            return SumOp(self.terms + other.terms)
        return SumOp(self.terms + [other])

    def __sub__(self, other):
        """
        Logical difference.
        """
        return self + (-other)

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        if not self.terms:
            # logical zero operator
            return "<empty sum>"
        s = ""
        for term in self.terms:
            s += ("" if s == "" else " + ") + str(term)
        return s

    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, SumOp):
            if len(self.terms) == len(other.terms):
                # assuming that terms are sorted
                if all(t1 == t2 for t1, t2 in zip(self.terms, other.terms)):
                    return True
        return False

    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if not isinstance(other, SumOp):
            return False
        # assuming that terms are sorted
        for t1, t2 in zip(self.terms, other.terms):
            if t1 < t2:
                return True
            if t2 < t1:
                return False
            assert t1 == t2
        if len(self.terms) < len(other.terms):
            return True
        if len(other.terms) < len(self.terms):
            return False
        # operators are equal
        return False

    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        # ignoring a potential global scalar factor, for simplicity
        return self == other

    def as_field_operator(self) -> FieldOp:
        """
        Represent the operator in terms of fermionic field operators.
        """
        s = FieldOp([])
        for term in self.terms:
            s += term.as_field_operator()
        return s

    def support(self) -> list[tuple]:
        """
        Support of the operator: lattice sites which it acts on (including spin).
        """
        s = []
        for term in self.terms:
            s += term.support()
        return sorted(list(set(s)))

    def translate(self, shift: Sequence[int]):
        """
        Translate by `shift` and return the translated operator.
        """
        return SumOp([term.translate(shift) for term in self.terms])

    def normalize(self) -> tuple:
        """
        Return a normalized copy of the operator together with its scalar prefactor.
        """
        return self, 1

    def is_zero(self) -> bool:
        """
        Indicate whether the operator acts as zero operator.
        """
        if not self.terms:
            return True
        # for simplicity, do not check here whether terms cancel
        return False

    @property
    def fermi_weight(self) -> int:
        """
        Maximum number of fermionic creation and annihilation operators multiplied together.
        """
        return max((term.fermi_weight for term in self.terms), default=0)

    def norm_bound(self) -> float:
        """
        Upper bound on the spectral norm of the operator.
        """
        nmodes = len(self.support())
        if nmodes <= HamiltonianOp.max_nmodes_exact_norm:
            # compute exact norm
            cmt = self.as_field_operator().as_compact_matrix()
            return _spectral_norm_conserved_particles(nmodes, cmt)

        if self.is_quadratic_sum():
            # compute exact norm based on quadratic form
            fop = self.as_field_operator()
            h = fop.quadratic_coefficients()
            if np.allclose(h, h.T):
                # single-particle eigenvalues
                levels = np.linalg.eigvalsh(h)
                en_pos = sum(max( en, 0) for en in levels)
                en_neg = sum(max(-en, 0) for en in levels)
                return max(en_pos, en_neg)
            if np.allclose(h, -h.T):
                # actual single-particle eigenvalues are purely imaginary
                # and come in conjugated pairs
                levels = np.linalg.eigvalsh(1j*h)
                return sum(max(en, 0) for en in levels)
            raise NotImplementedError

        # try to split into disjoint components
        supp = self.support()
        L = len(supp)
        adj = np.zeros((L, L), dtype=int)
        for term in self.terms:
            # assuming that operators in a term cannot be further partitioned
            slist = [supp.index(s) for s in term.support()]
            for i in slist:
                for j in slist:
                    if i != j:
                        adj[i, j] = 1
        # connected components
        components = _connected_components(adj)
        if len(components) >= 2:
            # bound is not equal to actual spectral norm in case leading
            # eigenvalues of operators on components have opposite sign
            nrm_bound = 0
            handled_terms = len(self.terms) * [False]
            for comp in components:
                clustered_terms = []
                for i, term in enumerate(self.terms):
                    if all(supp.index(s) in comp for s in term.support()):
                        clustered_terms.append(term)
                        handled_terms[i] = True
                    else:
                        assert not any(supp.index(s) in comp for s in term.support())
                nrm_bound += SumOp(clustered_terms).norm_bound()
            assert all(handled_terms)
            return nrm_bound

        # partitioning into clusters with bounded support
        nrm_bound = 0
        # partition terms into groups, such that each group has at most 'max_nmodes_exact_norm' support,
        # starting from end (summands with largest weight)
        remaining = list(reversed(range(len(self.terms))))
        supps = [set(term.support()) for term in self.terms]
        while remaining:
            i = remaining[0]
            remaining.remove(i)
            supp = supps[i]
            if len(supp) > HamiltonianOp.max_nmodes_exact_norm:
                warn(f"encountered individual summand supported on {len(supp)} > {HamiltonianOp.max_nmodes_exact_norm} modes")
                nrm_bound += self.terms[i].norm_bound()
                continue
            fop = self.terms[i].as_field_operator()
            while remaining:
                # select term with minimum additional modes
                diff = [len(supps[j].difference(supp)) for j in remaining]
                if len(supp) + min(diff) <= HamiltonianOp.max_nmodes_exact_norm:
                    j = remaining[diff.index(min(diff))]
                    assert len(supps[j].difference(supp)) == min(diff)
                    supp = supp.union(supps[j])
                    fop += self.terms[j].as_field_operator()
                    remaining.remove(j)
                    continue
                else:
                    break
            assert len(supp) <= HamiltonianOp.max_nmodes_exact_norm
            # compute exact norm
            cmt = fop.as_compact_matrix()
            nrm_bound += _spectral_norm_conserved_particles(len(supp), cmt)
        return nrm_bound

    def is_quadratic_sum(self) -> bool:
        """
        Whether the operator is a sum of quadratic field operators.
        """
        return all(isinstance(term, (HoppingOp, AntisymmHoppingOp, NumberOp)) for term in self.terms)

    def is_numop_sum(self) -> bool:
        """
        Whether the operator is a sum of number operators or products of number operators.
        """
        return bool(self.terms) and all( isinstance(term, NumberOp) or
                                        (isinstance(term, ProductOp) and term.is_numop_product())
                                            for term in self.terms)


def _spectral_norm_conserved_particles(nmodes: int, op):
    """
    Compute the spectral norm of 'op', assuming that 'op' conserves the
    fermionic particle number, and can hence be partitioned into blocks.
    """
    assert op.shape == 2 * (2**nmodes,)
    max_nrm = 0
    for n in range(nmodes):
        idx = [f for f in range(2**nmodes) if f.bit_count() == n]
        nrm = np.linalg.norm(op[:, idx][idx, :].todense(), ord=2)
        max_nrm = max(max_nrm, nrm)
    return max_nrm


def _find_component(adj, visited: list, i: int, comp: list):
    """
    Find the connected component (reachable vertices) starting from vertex `i`
    in a graph specified by the adjacency matrix `adj`.
    """
    n = len(adj)
    for j in range(n):
        if adj[i, j]:
            if not visited[j]:
                comp.append(j)
                visited[j] = True
                _find_component(adj, visited, j, comp)


def _connected_components(adj):
    """
    Determine the connected components of a graph specified by the adjacency matrix `adj`.
    """
    n = len(adj)
    visited = n * [False]
    components = []
    for i in range(n):
        if not visited[i]:
            visited[i] = True
            comp = [i]
            _find_component(adj, visited, i,  comp)
            components.append(comp)
    return components
