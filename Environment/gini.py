"""
    Calculate the global reward penalized using the gini_coeficient coefficient of a numpy array of data.
    We use this value to provide a reward that account for the better distribution of MOS among the stations
    It substitutes the baseline that uses the average of MOS

    based on the statsdirect equation shown in
        http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini_coeficient.htm

    Other references:
    * https://en.wikipedia.org/wiki/Gini_coefficient
    * https://towardsdatascience.com/gini_coeficient-coefficient-and-lorenz-curve-f19bb8f46d66
    * DORFMAN, Robert. A formula for the gini_coeficient coefficient. The review of economics and statistics, p. 146-149, 1979.

"""
import numpy as np


def scale_minmax(data):
    _min = np.amin(data)
    _max = np.amax(data)
    scaled = (data - _min)
    if _min != _max:
        scaled /= (_max - _min)
    return scaled


def gini_coeficient(user_data, epsilon=1e-18):
    """ Calculate the gini_coeficient coefficient of a numpy data.
        All values are treated equally,
        the values are first placed in ascending order, such that each x has rank i,

        @return: 0 if the data is homogeneous or 1 if the data
    """
    data = user_data.flatten()
    assert np.amin(data) >= 0, "Values cannot be negative"
    assert len(data.shape) == 1, "Data must be 1D"
    scaled = scale_minmax(data)

    # Sort data to calculate index
    data = np.sort(scaled)

    # Number of data elements:
    n = data.shape[0]
    # Index per data element:
    index = np.arange(1, n + 1)

    # gini_coeficient coefficient:
    coef_ = ((np.sum((2 * index - n - 1) * data)) / (n * np.sum(data) + epsilon))

    return coef_


def reward_gini(data):
    avg = np.average(data)
    g = gini_coeficient(data)
    r = (1 - g) * avg
    return r


if __name__ == "__main__":
    n = 20
    d = np.zeros(n)
    g = gini_coeficient(d)
    print("homogeneous zeros: {} ==> reward {}".format(g, reward_gini(d)))

    d[0] = 1
    g = gini_coeficient(d)
    print("heterogeneous - one 1: {} ==> reward {}".format(g, reward_gini(d)))

    d = np.full(n, 0.5)  # average MOS
    g = gini_coeficient(d)
    print("homogeneous 0.5: {} ==> reward {}".format(g, reward_gini(d)))

    d = np.ones(n)  # average MOS
    g = gini_coeficient(d)
    print("homogeneous 1: {} ==> reward {}".format(g, reward_gini(d)))
