import numpy as np
import numpy.fft as fft


class Order:
    """
    An object that defines the order in which the distance profiles are calculated for a given Matrix Profile
    """

    def next(self):
        raise NotImplementedError("next() not implemented")


class linearOrder(Order):
    """
    An object that defines a linear (iterative) order in which the distance profiles are calculated for a given Matrix Profile
    """

    def __init__(self, m):
        self.m = m
        self.idx = -1

    def next(self):
        """
        Advances the Order object to the next index
        """
        self.idx += 1
        if self.idx < self.m:
            return self.idx
        else:
            return None


def DotProductStomp(ts, m, dot_first, dot_prev, order):
    """
    Updates the sliding dot product for a time series ts from the previous dot product dot_prev.

    Parameters
    ----------
    ts: Time series under analysis.
    m: Length of query within sliding dot product.
    dot_first: The dot product between ts and the beginning query (QT1,1 in Zhu et.al).
    dot_prev: The dot product between ts and the query starting at index-1.
    order: The location of the first point in the query.
    """

    l = len(ts) - m + 1
    dot = np.roll(dot_prev, 1)

    dot += ts[order + m - 1] * ts[m - 1 : l + m] - ts[order - 1] * np.roll(ts[:l], 1)

    # Update the first value in the dot product array
    dot[0] = dot_first[order]

    return dot


def massStomp(query, ts, dot_first, dot_prev, index, mean, std, z_norm=True):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS) between a query and timeseries using the STOMP dot product speedup. Note that we are returning the square of MASS.

    Parameters
    ----------
    query: Time series snippet to evaluate. Note that, for STOMP, the query must be a subset of ts.
    ts: Time series to compare against query.
    dot_first: The dot product between ts and the beginning query (QT1,1 in Zhu et.al).
    dot_prev: The dot product between ts and the query starting at index-1.
    index: The location of the first point in the query.
    mean: Array containing the mean of every subsequence in ts.
    std: Array containing the standard deviation of every subsequence in ts.
    """
    m = len(query)
    dot = DotProductStomp(ts, m, dot_first, dot_prev, index)

    # Return both the MASS calcuation and the dot product
    if z_norm:
        res = 2 * m * (1 - (dot - m * mean[index] * mean) / (m * std[index] * std))
    else:
        q_zero_mean = query - mean[index]
        # Compute squared distance using convolution.
        squared_distance = (
            np.convolve(ts**2, np.ones(m), mode="valid")
            - 2 * np.convolve(ts, q_zero_mean[::-1], mode="valid")
            + np.sum(q_zero_mean**2)
        )
        # Adjust for the zero-meaning of the time series windows.
        # squared_distance += m * mean**2
        # squared_distance -= 2 * m * mean * mean[index]
        squared_distance -= m * mean**2
        res = squared_distance

    return res, dot


def slidingDotProduct(query, ts):
    """
    Calculate the dot product between a query and all subsequences of length(query) in the timeseries ts. Note that we use Numpy's rfft method instead of fft.

    Parameters
    ----------
    query: Specific time series query to evaluate.
    ts: Time series to calculate the query's sliding dot product against.
    """

    m = len(query)
    n = len(ts)

    # If length is odd, zero-pad time time series
    ts_add = 0
    if n % 2 == 1:
        ts = np.insert(ts, 0, 0)
        ts_add = 1

    q_add = 0
    # If length is odd, zero-pad query
    if m % 2 == 1:
        query = np.insert(query, 0, 0)
        q_add = 1

    # This reverses the array
    query = query[::-1]

    query = np.pad(query, (0, n - m + ts_add - q_add), "constant")

    # Determine trim length for dot product. Note that zero-padding of the query has no effect on array length, which is solely determined by the longest vector
    trim = m - 1 + ts_add

    dot_product = fft.irfft(fft.rfft(ts) * fft.rfft(query))

    # Note that we only care about the dot product results from index m-1 onwards, as the first few values aren't true dot products (due to the way the FFT works for dot products)
    return dot_product[trim:]


def movmeanstd(ts, m):
    """
    Calculate the mean and standard deviation within a moving window passing across a time series.

    Parameters
    ----------
    ts: Time series to evaluate.
    m: Width of the moving window.
    """
    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    # Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts), 0, 0)
    # Add zero to the beginning of the cumsum of ts ** 2
    sSq = np.insert(np.cumsum(ts**2), 0, 0)
    segSum = s[m:] - s[:-m]
    segSumSq = sSq[m:] - sSq[:-m]

    movmean = segSum / m
    movstd = np.sqrt(segSumSq / m - (segSum / m) ** 2)

    return [movmean, movstd]


def mass(query, ts, z_norm=True):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS): a Euclidian distance similarity search algorithm. Note that we are returning the square of MASS.

    Parameters
    ----------
    :query: Time series snippet to evaluate. Note that the query does not have to be a subset of ts.
    :ts: Time series to compare against query.
    """

    # query_normalized = zNormalize(np.copy(query))
    m = len(query)
    q_mean = np.mean(query)
    q_std = np.std(query)
    mean, std = movmeanstd(ts, m)
    dot = slidingDotProduct(query, ts)

    if z_norm:
        res = 2 * m * (1 - (dot - m * mean * q_mean) / (m * std * q_std))
    else:
        q_zero_mean = query - q_mean
        # Compute squared distance using convolution.
        squared_distance = (
            np.convolve(ts**2, np.ones(m), mode="valid")
            - 2 * np.convolve(ts, q_zero_mean[::-1], mode="valid")
            + np.sum(q_zero_mean**2)
        )
        # Adjust for the zero-meaning of the time series windows.
        # squared_distance += m * mean**2
        # squared_distance -= 2 * m * mean * q_mean
        squared_distance -= m * mean**2
        res = squared_distance

    return res


def STOMPDistanceProfile(tsA, idx, m, dot_first, dp, mean, std, z_norm=True):
    """
    Returns the distance profile of a query within tsA against the time series tsB using the even more efficient iterative STOMP calculation. Note that the method requires a pre-calculated 'initial' sliding dot product.

    Parameters
    ----------
    tsA: Time series containing the query for which to calculate the distance profile.
    idx: Starting location of the query within tsA
    m: Length of query.
    tsB: Time series to compare the query against. Note that, for the time being, only tsB = tsA is allowed
    dot_first: The 'initial' sliding dot product, or QT(1,1) in Zhu et.al
    dp: The dot product between tsA and the query starting at index m-1
    mean: Array containing the mean of every subsequence of length m in tsA (moving window)
    std: Array containing the mean of every subsequence of length m in tsA (moving window)
    """

    query = tsA[idx : (idx + m)]
    n = len(tsA)

    # Calculate the first distance profile via MASS
    if idx == 0:
        distanceProfile = np.real(np.sqrt(mass(query, tsA, z_norm).astype(complex)))

        # Currently re-calculating the dot product separately as opposed to updating all of the mass function...
        dot = slidingDotProduct(query, tsA)

    # Calculate all subsequent distance profiles using the STOMP dot product shortcut
    else:
        res, dot = massStomp(query, tsA, dot_first, dp, idx, mean, std, z_norm)
        distanceProfile = np.real(np.sqrt(res.astype(complex)))

    trivialMatchRange = (
        int(max(0, idx - np.round(m / 4, 0))),
        int(min(idx + np.round(m / 4 + 1, 0), n)),
    )
    distanceProfile[trivialMatchRange[0] : trivialMatchRange[1]] = np.inf

    # Both the distance profile and corresponding matrix profile index (which should just have the current index)
    return (distanceProfile[idx:], np.full(n - m + 1, idx, dtype=float)), dot


def _matrixProfile_stomp(
    tsA, m, orderClass, distanceProfileFunction, z_norm=True, reverse=False
):
    if reverse:
        tsA = tsA[::-1]  # Reverse the time series

    order = orderClass(len(tsA) - m + 1)
    n = len(tsA)
    shape = n - m + 1
    mp, mpIndex = (np.full(shape, np.inf), np.full(shape, np.inf))

    tsA = np.array(tsA)
    search = np.isinf(tsA) | np.isnan(tsA)
    tsA[search] = 0

    idx = order.next()

    # Get moving mean and standard deviation
    mean, std = movmeanstd(tsA, m)

    # Initialize code to set dot_prev to None for the first pass
    dp = None

    # Initialize dot_first to None for the first pass
    dot_first = None

    while idx != None:
        # Need to pass in the previous sliding dot product for subsequent distance profile calculations
        (distanceProfile, querySegmentsID), dot_prev = distanceProfileFunction(
            tsA, idx, m, dot_first, dp, mean, std, z_norm
        )

        if idx == 0:
            dot_first = dot_prev

        # Update only the current index using the minimum value from the distanceProfile
        min_idx = idx + np.argmin(distanceProfile)
        mp[idx] = np.min(distanceProfile)
        mpIndex[idx] = min_idx

        idx = order.next()

        dp = dot_prev
    if reverse: 
        mp = mp[::-1]
        mpIndex = n + m - 1 - mpIndex
    return (mp, mpIndex)


def stomp(tsA, m, z_norm=True, reverse=False):
    """
    Calculate the Matrix Profile using the more efficient MASS calculation. Distance profiles are computed
    according to the directed STOMP procedure. If reverse is True, calculations are performed on the reversed
    time series, but the indices in the profile will refer to the original series.

    Parameters
    ----------
    tsA: Time series containing the queries for which to calculate the Matrix Profile.
    m: Length of subsequence to compare.
    reverse: If True, perform the STOMP on the reversed time series.
    """
    return _matrixProfile_stomp(
        tsA, m, linearOrder, STOMPDistanceProfile, z_norm, reverse
    )
