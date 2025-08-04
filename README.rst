Trotter error with commutator scaling for the Fermi-Hubbard model
=================================================================

.. image:: https://github.com/qc-tum/fermi_hubbard_commutators/actions/workflows/ci.yml/badge.svg
  :target: https://github.com/qc-tum/fermi_hubbard_commutators/actions/workflows/ci.yml


Python package for evaluating nested commutators between fermionic operators and computing Trotter splitting error bounds based on commutator scaling for the Fermi-Hubbard model, accompanying

| Ansgar Schubert, Christian B. Mendl
| *Trotter error with commutator scaling for the Fermi-Hubbard model*
| `Phys. Rev. B 108, 195105 (2023) <https://doi.org/10.1103/PhysRevB.108.195105>`_ (`arXiv:2306.10603 <https://arxiv.org/abs/2306.10603>`_)


Documentation
-------------
The following tutorials provide an introduction to the main concepts and features of *fh_comm*:

- `Hamiltonian operators <doc/hamiltonian_operators.ipynb>`_
- `Lattice translations <doc/lattice_translations.ipynb>`_
- `Splitting methods <doc/splitting_methods.ipynb>`_
- `Commutator bounds <doc/commutator_bounds.ipynb>`_

The full documentation is available at `fermi-hubbard-commutators.readthedocs.io <https://fermi-hubbard-commutators.readthedocs.io>`_.


Installation
------------
To install the *fh_comm* package, clone this repository and install it in development mode via

.. code-block:: python

    python3 -m pip install -e <path/to/repo>
