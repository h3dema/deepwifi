"""
    The result ranges from 1/n (worst case) to 1 (best case), and
    it is maximum when all users receive the same allocation.

    References:
    * https://en.wikipedia.org/wiki/Fairness_measure

"""
import numpy as np


def fairness_index(data, epsilon=1e-18):
    """

        @return: the jain fairness index, bounded between 0 and 1
                0 means the data is homogeneous (all values are equal) and
                1 means the data is different
    """
    assert np.amin(data) >= 0, "Values cannot be negative"
    assert np.amax(data) <= 1, "Values should be bounded between 0 and 1"
    assert len(data.shape) == 1, "Data must be 1D"

    # test if all values are maximum
    num = np.square(np.average(data))
    denom = np.average(np.square(data))
    f = 1 if denom == 0 else num / denom
    return f


def reward_jain(data):
    avg = np.average(data)
    g = fairness_index(data)
    r = g * avg
    return r


if __name__ == "__main__":
    n = 20
    d = np.zeros(n)
    g = fairness_index(d)
    print("homogeneous zeros: {} ==> reward {}".format(g, reward_jain(d)))

    d[0] = 1
    g = fairness_index(d)
    print("heterogeneous - one 1: {} ==> reward {}".format(g, reward_jain(d)))

    d[1] = 1
    g = fairness_index(d)
    print("heterogeneous - two 1: {} ==> reward {}".format(g, reward_jain(d)))

    d = np.ones(n)
    d[0] = 0
    g = fairness_index(d)
    print("heterogeneous - one 0: {} ==> reward {}".format(g, reward_jain(d)))

    d = np.full(n, 0.5)  # average MOS
    g = fairness_index(d)
    print("homogeneous 0.5: {} ==> reward {}".format(g, reward_jain(d)))

    d = np.ones(n)  # average MOS
    g = fairness_index(d)
    print("homogeneous 1: {} ==> reward {}".format(g, reward_jain(d)))
