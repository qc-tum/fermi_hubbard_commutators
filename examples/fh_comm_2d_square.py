import numpy as np
import fh_comm as fhc


def main():

    # Hamiltonian coefficients
    v = -1
    u =  1  # other values can be included as prefactor depending on the number of interaction terms

    # construct a sub-lattice for translations
    translatt = fhc.SubLattice(np.array([[1, 1], [1, -1]]))

    # construct the Fermi-Hubbard Hamiltonian operators on a 2D square lattice
    h0 = fhc.SumOp([fhc.HoppingOp(( 0,  0), ( 1,  0), s, v) for s in [0, 1]])
    h1 = fhc.SumOp([fhc.HoppingOp((-1,  0), ( 0,  0), s, v) for s in [0, 1]])
    h2 = fhc.SumOp([fhc.HoppingOp(( 0,  0), ( 0,  1), s, v) for s in [0, 1]])
    h3 = fhc.SumOp([fhc.HoppingOp(( 0, -1), ( 0,  0), s, v) for s in [0, 1]])
    h4 = fhc.SumOp([fhc.ProductOp([fhc.NumberOp((x, 0), s, 1) for s in [0, 1]], u) for x in [0, 1]])
    hlist = [h0, h1, h2, h3, h4]

    # example
    print("[h0, h1]:", fhc.commutator_translation(h0, h1, translatt))
    print("[h0, h4]:", fhc.commutator_translation(h0, h4, translatt))

    comm_tab = fhc.NestedCommutatorTable(hlist, 5, translatt, bias=1e-8)
    tab1 = comm_tab.table(1)
    tab2 = comm_tab.table(2)
    tab4 = comm_tab.table(4)
    # example
    print("tab1[0][1]:", tab1[0][1])
    # example
    print("tab2[0][4][1]:", tab2[0][4][1])
    print("tab2[0][4][1].norm_bound():", tab2[0][4][1].norm_bound())
    # example
    print("tab2[1][3][4]:", tab2[1][3][4])
    print("tab2[1][3][4].norm_bound():", tab2[1][3][4].norm_bound())
    # example
    print("tab4[0][4][4][4][4]:", tab4[0][4][4][4][4])
    print("tab4[0][4][4][4][4].norm_bound():", tab4[0][4][4][4][4].norm_bound())

    for methname in ["Strang", "Suzuki4"]:
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


if __name__ == "__main__":
    main()
