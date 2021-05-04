def find_nearest(array, values):
    '''
    Find the index of the values in the array closest to the given value
    ------
    Parameters:
    array: numpy.ndarray
        numpy array to be matched
    values: numpy.ndarray or float64
        values to be matched with numpy array
    ------
    Return:
    idx: int
        index of the closest value in array.
    '''
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array[:,np.newaxis] - values)), axis=0)

    return idx

def array_isin(element, test_elements):
    '''
    Test if element in one array is in another array
    ------
    Parameters
    element: np.ndarray
        Input array
    test_elements: np.ndarray
        The values against which to test each value of element. This 
        argument is flattened if it is an array or array_like. 
    ------
    Return
    isin: np.ndarray
        Indexes of element that in test_elements. 
    '''
    dist = np.abs(element[:, np.newaxis] - test_elements)
    diff = np.nanmin(dist, axis=1)
    isin = np.where(diff == 0)

    return isin


def regrid_time(xdata, ydata, xtime, ytime,
                interval=1.0):
    '''
    ------
    paramters:
    xdata: np.ndarray.
        array of data to be synchronized to
    ydata: np.ndarray
        array of data to be synchronized
    xtime: np.ndarray.
        array of time the xdata is taken
    ytime: np.ndarray.
        array of time the ydata is taken
    interval: float
        maximal alllowd time difference between matched two time
    series. 
    ------
    retrun: 
    ydata_regridded: np.array
        regridded ydata with same shape as xdata
    '''
    ydata_regridded = np.full(np.shape(xdata), np.nan)

    # match the xtime with ytime 
    dist = np.abs(ytime[:, np.newaxis] - xtime)
    potentialClosest = dist.argmin(axis=1)
    diff = dist.min(axis=1)
    # leave out the time with spacing greater than the interval of original time.
    ydata[np.where(diff > interval)] = np.nan
    closestFound, closestCounts = np.unique(potentialClosest, return_counts=True)
    ydata_group = np.split(ydata, np.cumsum(closestCounts)[:-1])

    for i, index in enumerate(closestFound):
        ydata_regridded[index] = np.nanmean(ydata_group[i])

    return ydata_regridded


def smooth_data(data, time, timebin=30):
    '''
    Smooth the data by every certain length of time. Note the length of smoothed data
    is the same as the original data. 
    ------
    parameters:
    data: np.ndarray
        array of data to be smoothed
    time: np.ndarray
        time series of the data taken
    timbine: float64
        The time used to average the data
    ------
    return:
    data_smooth: np.ndarray
        smoothed data
    '''
    data_smooth = np.full(np.shape(time), np.nan)
    for i in range(np.shape(time)[0]):
        conditions = ~((time >= time[i]) & (time <= (time[i]+30)))
        data_temp = np.copy(data)
        data_temp[np.where(conditions)] = np.nan
        data_smooth[i] = np.nanmean(data_temp)

    return data_smooth

