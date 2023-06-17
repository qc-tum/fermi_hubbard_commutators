from collections.abc import Sequence
from .lattice import SubLattice
from .hamiltonian_ops import HamiltonianOp
from .commutator import commutator, commutator_translation
from .simplification import simplify, translate_origin


class NestedCommutatorTable:
    """
    Evaluate all nested commutators up to a specified depth.
    """
    def __init__(self, hlist: Sequence[HamiltonianOp], depth: int, translatt: SubLattice = None, bias: float = 0):
        # build the table for all levels
        if depth < 1:
            raise ValueError("'depth' must at least be 1")
        self._table = [list(hlist)]
        for d in range(1, depth):
            self._table.append(_nested_commutators(hlist, self._table[d - 1], translatt, bias))

    def table(self, level: int):
        """
        Get the table at level `level`.
        """
        return self._table[level]


def _nested_commutators(hlist: Sequence[HamiltonianOp], operators, translatt: SubLattice, bias: float):
    """
    Replace the innermost level of `operators` with a list of commutators
    between the entries of `hlist` and the current operator.
    """
    if isinstance(operators, list):
        return [_nested_commutators(hlist, op, translatt, bias) for op in operators]
    if translatt is None:
        return [simplify(commutator(h, operators)) for h in hlist]
    return [simplify(translate_origin(
        simplify(commutator_translation(h, operators, translatt)),
        translatt,
        bias)) for h in hlist]
