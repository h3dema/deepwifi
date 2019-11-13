"""
    The hossfeld index ranges from 0 (worst case) to 1 (best case), and
    it is maximum when all users receive the same allocation (homogeneity).

    References:
    * https://en.wikipedia.org/wiki/Fairness_measure

"""
import numpy as np
from copy import deepcopy


def hossfeld_index(data, L=1, H=5):
    """
        1 indicating perfect QoE fairness - all users experience the same quality.
        0 indicates total unfairness, e.g. 50% of users experience highest QoE H and 50% experience lowest QoE L.

        @param L: lower bound in data, for MOS = 1
        @param H: upper bound in data, for MOS = 5
        @return: the Hostfeld fairness index, bounded between 0 and 1
    """
    assert np.amin(data) >= L and np.amax(data) <= H, "Values should be bounded between {} and {}".format(L, H)
    assert len(data.shape) == 1, "Data must be 1D"

    F = 1 - 2 * np.std(data) / (H - L)
    return F


def reward_hossfeld(data, C=0.4):
    """ gets a compromise between the average of the reward and the Hossfeld Index

        @param data: array with MOS values
        @return: the reward for each entry
    """
    values = deepcopy(data)
    avg = np.average(values)
    h = hossfeld_index(values)
    r = 1 + (avg - 1) * (h ** C)
    # set to average only the data above average
    items_to_trim = values > avg
    values[items_to_trim] = r  # above average are penalyzed
    values[~items_to_trim] = avg  # below average go to average
    return values


if __name__ == "__main__":
    n = 20
    d = np.zeros(n)
    g = hossfeld_index(d)
    print("homogeneous zeros: {} ==> reward {}".format(g, reward_hossfeld(d)))

    d[0] = 1
    g = hossfeld_index(d)
    print("heterogeneous - one 1: {} ==> reward {}".format(g, reward_hossfeld(d)))

    d[1] = 1
    g = hossfeld_index(d)
    print("heterogeneous - two 1: {} ==> reward {}".format(g, reward_hossfeld(d)))

    d = np.ones(n)
    d[0] = 0
    g = hossfeld_index(d)
    print("heterogeneous - one 0: {} ==> reward {}".format(g, reward_hossfeld(d)))

    d = np.full(n, 0.5)  # average MOS
    g = hossfeld_index(d)
    print("homogeneous 0.5: {} ==> reward {}".format(g, reward_hossfeld(d)))

    d = np.ones(n)  # average MOS
    g = hossfeld_index(d)
    print("homogeneous 1: {} ==> reward {}".format(g, reward_hossfeld(d)))
