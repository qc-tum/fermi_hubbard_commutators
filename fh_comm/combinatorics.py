from scipy.special import binom


def multinomial(params):
    """
    Multinomial coefficient.
    """
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])


def integer_sum_tuples(s: int, nbins: int):
    """
    Generate lexicographically sorted non-negative integer lists
    such that the sum of the integers equals 's'.
    """
    if nbins <= 0:
        raise ValueError(f"'nbins' must be at least 1, received {nbins}")
    if s < 0:
        raise ValueError(f"'s' cannot be negative, received {s}")
    c = nbins * [0]
    c[-1] = s
    yield tuple(c)
    done = False
    while not done:
        done = True
        for i in reversed(range(1, nbins)):
            if c[i] > 0:
                c[i] -= 1
                # swap c[-1] <-> c[i]
                c[-1], c[i] = c[i], c[-1]
                c[i - 1] += 1
                yield tuple(c)
                done = False
                break
