import numpy as np
import scipy
import matplotlib.pyplot as plt
import fh_comm as fhc


def trotterized_time_evolution(hlist, method: fhc.SplittingMethod, dt: float, nsteps: int):
    """
    Compute the numeric ODE flow operator of the quantum time evolution
    based on the provided splitting method.
    """
    V = None
    for i, c in zip(method.indices, method.coeffs):
        if V is None:
            V = scipy.linalg.expm(-1j*c*dt*hlist[i])
        else:
            V = scipy.linalg.expm(-1j*c*dt*hlist[i]) @ V
    return np.linalg.matrix_power(V, nsteps)


def main():

    # Hamiltonian coefficients
    v = -1
    u =  1  # other values can be included as prefactor depending on the number of interaction terms

    # construct a sub-lattice for translations
    translatt = fhc.SubLattice(np.array([[2]]))

    # construct the Fermi-Hubbard Hamiltonian operators on a 1D lattice
    h0 = fhc.SumOp([fhc.HoppingOp(( 0,), ( 1,), s, v) for s in [0, 1]])
    h1 = fhc.SumOp([fhc.HoppingOp((-1,), ( 0,), s, v) for s in [0, 1]])
    h2 = fhc.SumOp([fhc.ProductOp([fhc.NumberOp((x,), s, 1) for s in [0, 1]], u) for x in [0, 1]])
    hlist = [h0, h1, h2]

    # matrix representation of Hamiltonian
    L = 4
    hlist_mat = [h.as_field_operator().as_matrix((L,), translatt).todense() for h in hlist]
    Hmat = sum(hlist_mat)

    comm_tab = fhc.NestedCommutatorTable(hlist, 5, translatt, bias=1e-8)
    tab2 = comm_tab.table(2)
    tab4 = comm_tab.table(4)
    # example
    print("tab2[1][2][1]:", tab2[1][2][1])
    print("tab2[1][2][1].norm_bound():", tab2[1][2][1].norm_bound())
    # example
    print("tab2[1][2][2]:", tab2[1][2][2])
    print("tab2[1][2][2].norm_bound():", tab2[1][2][2].norm_bound())
    # example
    print("tab2[0][2][1]:", tab2[0][2][1])
    print("tab2[0][2][1].norm_bound():", tab2[0][2][1].norm_bound())
    # example
    print("tab4[1][2][2][0][2]:", tab4[1][2][2][0][2])
    print("tab4[1][2][2][0][2].norm_bound():", tab4[1][2][2][0][2].norm_bound())

    for imeth, methname in enumerate(["Strang", "Suzuki4", "AK 11-4"]):
        print(80 * "_")
        print("Method:", methname)
        if methname == "Strang":
            # Strang splitting method
            method = fhc.SplittingMethod.suzuki(len(hlist), 1)
            print(method)
            # tight bound
            print("bound specialized for second-order Suzuki method:")
            comm_bound_terms = fhc.commutator_bound_strang(len(hlist))
        elif methname == "Suzuki4":
            method = fhc.SplittingMethod.suzuki(len(hlist), 2)
            print(method)
            s = (method.num_layers + 1) // 2
            print("s:", s)
            comm_bound_terms = fhc.commutator_bound(method, s)
        elif methname == "AK 11-4":
            method = fhc.SplittingMethod.ak_11_4()
            print(method)
            s = (method.num_layers + 1) // 2
            print("s:", s)
            comm_bound_terms = fhc.commutator_bound(method, s)
        # sort by number of interaction terms
        err_bound = (method.order + 1) * [0]
        for term in comm_bound_terms:
            print(term)
            num_int = sum(1 if i == len(hlist)-1 else 0 for i in term.commidx)
            if method.order == 2:
                err_bound[num_int] += term.weight * tab2[term.commidx[0]][term.commidx[1]][term.commidx[2]].norm_bound()
            elif method.order == 4:
                err_bound[num_int] += term.weight * tab4[term.commidx[0]][term.commidx[1]][term.commidx[2]][term.commidx[3]][term.commidx[4]].norm_bound()
        print("err_bound:", err_bound)

        tlist = [0.5**n for n in range(8)]
        err_ref = np.zeros(len(tlist))
        errcomm = np.zeros(len(tlist))
        for i, t in enumerate(tlist):
            # reference global unitary
            expitH = scipy.linalg.expm(-1j*Hmat*t)
            V = trotterized_time_evolution(hlist_mat, method, t, 1)
            # empirical error per lattice site
            err_ref[i] = np.linalg.norm(V - expitH, ord=2) / L
            # factor 1/2 to get the error per lattice site (terms are understood as translations by two sites)
            errcomm[i] = t**(method.order + 1) * 0.5 * sum(err_bound)
        # visualize results
        markers = ['.', '*', '^']
        plt.loglog(tlist, err_ref, markers[imeth]+'-', label=f"empir. error (L = {L}), " + methname, color="C"+str(imeth))
        plt.loglog(tlist, errcomm, markers[imeth]+'--', label="comm. scaling, " + methname, color="C"+str(imeth))
    plt.xlabel("t")
    plt.ylabel("error")
    plt.title(f"Splitting for Fermi-Hubbard on a 1D lattice, v = {v}, u = {u}")
    plt.legend()
    plt.savefig("fh_comm_1d_error.pdf")
    plt.show()


if __name__ == "__main__":
    main()
