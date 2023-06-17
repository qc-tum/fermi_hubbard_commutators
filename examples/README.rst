Trotter error bounds for the Fermi-Hubbard model on various lattice geometries
------------------------------------------------------------------------------

``comm_bound_demo.py`` showcases the bounds with small prefactors for various splitting methods.

The ``fh_comm_<geometry>.py`` script evaluates the commutator bounds for the Fermi-Hubbard model on the respective lattice geometry.
The calculations for the fourth-order methods take quite long (hours to days)!
You can restrict the computations to the Strang (second-order Suzuki) method by constructing the ``NestedCommutatorTable`` with depth 3 (instead of 5) and commenting out the level-4 commutator table.
Alternatively, setting ``HamiltonianOp.max_nmodes_exact_norm`` to a smaller value (like 12 or 10) will also shorten the computation but lead to slightly weaker bounds.
