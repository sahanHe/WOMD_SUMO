import numpy as np

def accel(std: float = 0.2):
    """
    - Definition: the desired acceleration of the vehicle
    - Default: ?
    - Range: >0
    """
    sample = np.random.normal(2, std)
    return min(max(sample, 1), 3)


def decel(mean: float = 3, std: float = 0.2):
    """
    - Definition: the desired deceleration of the vechile
    - Default: ?
    - Range: >0
    """
    sample = np.random.normal(loc=mean, scale=std)
    return min(max(sample, 1), 4.5)


def mingap():
    """
    - Definition: the minimum longitude gap
    - Default: 2.5
    - Range: [0, inf)
    """
    sample = np.random.normal(loc=2.5, scale=0.5)
    return min(max(sample, 0), 5)


def sigma():
    """
    - Definition: driver imperfection
    - Default: 0.5
    - Range: [0,1]
    """
    sample = np.random.normal(loc=0.5, scale=0.2)
    return min(max(sample, 0), 1)


def tau():
    """
    - Definition: desired time headway
    - Default: 1
    - Range: [0, inf)
    """
    sample = np.random.lognormal(mean=0, sigma=0.1)
    return min(max(sample, 0), 5)


def startupDelay():
    """
    - Definition: delay to start when traffic signal turns green
    - Default: 0
    - Range: [0, inf)
    """
    sample = np.random.exponential(scale=0.3)
    return min(sample, 1)


def minGapLat():
    """
    - Definition: the minimum latitude gap
    - Default: 0.6
    - Range: [0, inf)
    """
    sample = np.random.normal(loc=0.6, scale=0.08)
    return min(max(sample, 0.4), 0.8)


def lc_strategic():
    """
    - Definition: The eagerness for performing strategic lane changing
    - Default: 1
    - Range: [0, inf)
    """
    sample = np.random.lognormal(mean=100, sigma=0.1)
    return max(sample, 1.5)


def lc_keepright():
    """
    - Definition: The eagerness for following the obligation to keep right.
    - Default:
    - Range:
    """
    sample = np.random.lognormal(mean=100, sigma=0.1)
    return max(sample, 1.5)


def lc_sublane():
    """
    - Definition: the eagerness for using the configured lateral alignment within the lane.
    - Default: 1
    - Range: [0, inf)
    """
    sample = np.random.normal(loc=0.4, scale=0.3)
    return min(max(sample, 0), 10)


def jm_stoplinegap():
    """
    - Definition: This value configures stopping distance in front of prioritary / TL-controlled stop line.
    - Default:
    - Range:
    """
    sample = np.random.lognormal(mean=0.4, sigma=0.5)
    return min(max(sample, 0), 3) + 1


def jm_ignorekeepcleartime():
    """
    - Definition:
    - Default:
    - Range:
    """
    a = np.random.random()
    if a < 0.9:
        return -1
    else:
        sample = np.random.normal(loc=6, scale=1)
        return min(max(sample, 0), 10)
