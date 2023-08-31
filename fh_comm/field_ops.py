import math
import enum
from numbers import Rational
from collections.abc import Sequence
from functools import cache
import numpy as np
from scipy import sparse
from fh_comm.lattice import latt_coord_to_index, SubLattice


class FieldOpType(enum.Enum):
    """
    Fermionic field operator type.
    """
    FERMI_CREATE  = 0   # fermionic creation operator
    FERMI_ANNIHIL = 1   # fermionic annihilation operator
    FERMI_NUMBER  = 2   # fermionic number operator


class ElementaryFieldOp:
    """
    Elementary fermionic field operator.
    """
    def __init__(self, otype: FieldOpType, i: Sequence[int], s: int):
        self.otype = otype
        self.i = tuple(i)
        assert s in [0, 1]
        self.s = s

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        if self.otype == FieldOpType.FERMI_CREATE:
            return f"ad_{{{self.i}, {'up' if self.s == 0 else 'dn'}}}"
        if self.otype == FieldOpType.FERMI_ANNIHIL:
            return f"a_{{{self.i}, {'up' if self.s == 0 else 'dn'}}}"
        if self.otype == FieldOpType.FERMI_NUMBER:
            return f"n_{{{self.i}, {'up' if self.s == 0 else 'dn'}}}"
        assert False


class ProductFieldOp:
    """
    Product of elementary fermionic field operators.
    """
    def __init__(self, ops: Sequence[ElementaryFieldOp], coeff: float):
        self.ops = list(ops)
        self.coeff = coeff

    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return ProductFieldOp(self.ops, other * self.coeff)

    def __matmul__(self, other):
        """
        Logical product.
        """
        return ProductFieldOp(self.ops + other.ops, self.coeff * other.coeff)

    def __neg__(self):
        """
        Logical negation.
        """
        return ProductFieldOp(self.ops, -self.coeff)

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        c = "" if self.coeff == 1 else f"({self.coeff}) "
        if not self.ops:
            # logical identity operator
            s = "id"
        else:
            s = ""
            for op in self.ops:
                s += ("" if s == "" else " ") + str(op)
        return c + s


class FieldOp:
    """
    Sum of products of fermionic field operators.
    """
    def __init__(self, terms: Sequence[ProductFieldOp]):
        self.terms = list(terms)

    def as_matrix(self, latt_shape: Sequence[int], translatt: SubLattice = None):
        """
        Generate the sparse matrix representation of the operator
        embedded in a square lattice with dimensions `latt_shape`
        and periodic boundary conditions.
        Optionally using shifted copies on sublattice `translatt`.
        """
        # number of lattice sites; factor 2 from spin
        L = 2 * math.prod(latt_shape)
        if not self.terms:
            # fast-return zero matrix if terms are empty
            return sparse.csr_matrix((2**L, 2**L))
        clist, alist, nlist = construct_fermionic_operators(L)
        mat = 0
        if translatt is None:
            tps = [len(latt_shape) * (0,)]
        else:
            tps = translatt.instantiate(len(latt_shape) * (0,), latt_shape)
        for tp in tps:
            for term in self.terms:
                fstring = sparse.identity(2**L)
                for op in term.ops:
                    # take spin into account for indexing
                    j = 2 * latt_coord_to_index(tuple(x + y for x, y in zip(op.i, tp)), latt_shape) + op.s
                    if op.otype == FieldOpType.FERMI_CREATE:
                        fstring = fstring @ clist[j]
                    elif op.otype == FieldOpType.FERMI_ANNIHIL:
                        fstring = fstring @ alist[j]
                    elif op.otype == FieldOpType.FERMI_NUMBER:
                        fstring = fstring @ nlist[j]
                    else:
                        raise RuntimeError(f"unexpected fermionic operator type {op.otype}")
                mat += float(term.coeff) * fstring
        return mat

    def support(self) -> list[tuple]:
        """
        Support of the operator: lattice sites which it acts on (including spin).
        """
        s = []
        for term in self.terms:
            for op in term.ops:
                s.append(op.i + (op.s,))
        return sorted(list(set(s)))

    def as_compact_matrix(self):
        """
        Generate the sparse matrix representation on a virtual lattice
        consisting of the sites acted on by the field operators.
        """
        if not self.terms:
            # fast-return zero matrix if terms are empty
            return sparse.csr_matrix((1, 1))
        # active sites (including spin)
        supp = self.support()
        L = len(supp)
        clist, alist, nlist = construct_fermionic_operators(L)
        # construct matrix representation
        mat = 0
        for term in self.terms:
            fstring = sparse.identity(2**L)
            for op in term.ops:
                # take spin into account for indexing
                j = supp.index(op.i + (op.s,))
                if op.otype == FieldOpType.FERMI_CREATE:
                    fstring = fstring @ clist[j]
                elif op.otype == FieldOpType.FERMI_ANNIHIL:
                    fstring = fstring @ alist[j]
                elif op.otype == FieldOpType.FERMI_NUMBER:
                    fstring = fstring @ nlist[j]
                else:
                    raise RuntimeError(f"unexpected fermionic operator type {op.otype}")
            mat += float(term.coeff) * fstring
        return mat

    def quadratic_coefficients(self):
        r"""
        Find the coefficients in the representation
        :math:`\sum_{i,j} h_{ij} a^{\dagger}_i a_j`,
        assuming that the field operator actually has this form.
        """
        if not self.terms:
            # fast-return zero matrix if terms are empty
            return np.zeros((1, 1))
        # active sites (including spin)
        supp = self.support()
        L = len(supp)
        h = np.zeros((L, L))
        for term in self.terms:
            if len(term.ops) == 1:
                op = term.ops[0]
                if op.otype != FieldOpType.FERMI_NUMBER:
                    raise ValueError("expecting number operator")
                j = supp.index(op.i + (op.s,))
                h[j, j] += term.coeff
            elif len(term.ops) == 2:
                op_a = term.ops[0]
                op_b = term.ops[1]
                if op_a.otype != FieldOpType.FERMI_CREATE:
                    raise ValueError("expecting creation operator")
                if op_b.otype != FieldOpType.FERMI_ANNIHIL:
                    raise ValueError("expecting annihilation operator")
                i = supp.index(op_a.i + (op_a.s,))
                j = supp.index(op_b.i + (op_b.s,))
                h[i, j] += term.coeff
            else:
                raise ValueError("field operator not of expected form")
        return h

    def __add__(self, other):
        """
        Logical sum.
        """
        return FieldOp(self.terms + other.terms)

    def __sub__(self, other):
        """
        Logical difference.
        """
        return FieldOp(self.terms + [-term for term in other.terms])

    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (Rational, float)):
            raise ValueError("expecting a scalar argument")
        return FieldOp([other * term for term in self.terms])

    def __matmul__(self, other):
        """
        Logical product.
        """
        # take all pairwise products
        return FieldOp([t1 @ t2 for t1 in self.terms for t2 in other.terms])

    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        if not self.terms:
            # logical zero operator
            return "<empty FieldOp>"
        s = ""
        for term in self.terms:
            s += ("" if s == "" else " + ") + str(term)
        return s


@cache
def construct_fermionic_operators(nmodes: int):
    """
    Generate sparse matrix representations of the fermionic creation and
    annihilation operators for `nmodes` modes (or sites),
    based on Jordan-Wigner transformation.
    """
    I = sparse.identity(2)
    Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    U = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    clist = []
    for i in range(nmodes):
        c = sparse.identity(1)
        for j in range(nmodes):
            if j < i:
                c = sparse.kron(c, I)
            elif j == i:
                c = sparse.kron(c, U)
            else:
                c = sparse.kron(c, Z)
        c = sparse.csr_matrix(c)
        c.eliminate_zeros()
        clist.append(c)
    # corresponding annihilation operators
    alist = [sparse.csr_matrix(c.conj().T) for c in clist]
    # corresponding number operators
    nlist = []
    for i in range(nmodes):
        f = 1 << (nmodes - i - 1)
        data = [1. if (n & f == f) else 0. for n in range(2**nmodes)]
        nlist.append(sparse.dia_matrix((data, 0), 2*(2**nmodes,)))
    return clist, alist, nlist
