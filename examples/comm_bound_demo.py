import fh_comm as fhc


def main():

    # Lie-Trotter splitting rule with two Hamiltonian terms
    rule = fhc.SplittingMethod.trotter(2)
    print(rule)
    s = (rule.num_layers + 1) // 2
    print("s:", s)
    terms = fhc.commutator_bound(rule, s)
    for term in terms:
        print(term)
    print()

    # Lie-Trotter splitting rule with three Hamiltonian terms
    rule = fhc.SplittingMethod.trotter(3)
    print(rule)
    s = (rule.num_layers + 1) // 2
    print("s:", s)
    terms = fhc.commutator_bound(rule, s)
    for term in terms:
        print(term)
    print()

    # Strang splitting rule with two Hamiltonian terms
    rule = fhc.SplittingMethod.suzuki(2, 1)
    print(rule)
    s = (rule.num_layers + 1) // 2
    print("s:", s)
    terms = fhc.commutator_bound(rule, s)
    for term in terms:
        print(term)
    terms_ref = terms
    print()

    # Strang splitting rule with two Hamiltonian terms, non-optimal 's'
    rule = fhc.SplittingMethod.suzuki(2, 1)
    print(rule)
    s = 1
    print("s:", s, "(non-optimal)")
    terms = fhc.commutator_bound(rule, s)
    for term, term_ref in zip(terms, terms_ref):
        assert term.commidx == term_ref.commidx
        print(term, "diff:", term.weight - term_ref.weight)
    print()

    # Strang splitting rule with three Hamiltonian terms
    rule = fhc.SplittingMethod.suzuki(3, 1)
    print(rule)
    s = (rule.num_layers + 1) // 2
    print("s:", s)
    terms = fhc.commutator_bound(rule, s)
    for term in terms:
        print(term)
    print()
    print("commutator bound specialized for Strang (second-order Suzuki) splitting rule:")
    terms = fhc.commutator_bound_strang(3)
    for term in terms:
        print(term)
    print()

    # fourth-order product rule with two Hamiltonian terms
    rule = fhc.SplittingMethod.suzuki(2, 2)
    print(rule)
    s = (rule.num_layers + 1) // 2
    print("s:", s)
    terms = fhc.commutator_bound(rule, s)
    for term in terms:
        print(term)
    terms_ref = terms
    print()

    # fourth-order product rule with two Hamiltonian terms, 's' not centered
    rule = fhc.SplittingMethod.suzuki(2, 2)
    print(rule)
    s = (rule.num_layers + 1) // 2 - 1
    print("s:", s, "(not centered)")
    terms = fhc.commutator_bound(rule, s)
    for term, term_ref in zip(terms, terms_ref):
        assert term.commidx == term_ref.commidx
        print(term, "diff:", term.weight - term_ref.weight)
    print()

    # fourth-order product rule with two Hamiltonian terms, 's' not centered
    rule = fhc.SplittingMethod.suzuki(2, 2)
    print(rule)
    s = (rule.num_layers + 1) // 2 + 1
    print("s:", s, "(not centered)")
    terms = fhc.commutator_bound(rule, s)
    for term, term_ref in zip(terms, terms_ref):
        assert term.commidx == term_ref.commidx
        print(term, "diff:", term.weight - term_ref.weight)
    print()

    # fourth-order product rule with two Hamiltonian terms, 's' not centered
    rule = fhc.SplittingMethod.suzuki(2, 2)
    print(rule)
    s = (rule.num_layers + 1) // 2 + 2
    print("s:", s, "(not centered)")
    terms = fhc.commutator_bound(rule, s)
    for term, term_ref in zip(terms, terms_ref):
        assert term.commidx == term_ref.commidx
        print(term, "diff:", term.weight - term_ref.weight)
    print()

    # fourth-order product rule with three Hamiltonian terms
    rule = fhc.SplittingMethod.suzuki(3, 2)
    print(rule)
    s = (rule.num_layers + 1) // 2
    print("s:", s)
    terms = fhc.commutator_bound(rule, s)
    for term in terms:
        print(term)
    terms_ref = terms
    print()

    # fourth-order product rule with three Hamiltonian terms, non-optimal 's'
    rule = fhc.SplittingMethod.suzuki(3, 2)
    print(rule)
    s = (rule.num_layers + 1) // 2 - 1
    print("s:", s, "(non-optimal)")
    terms = fhc.commutator_bound(rule, s)
    for term, term_ref in zip(terms, terms_ref):
        assert term.commidx == term_ref.commidx
        print(term, "diff:", term.weight - term_ref.weight)
    print()


if __name__ == "__main__":
    main()
