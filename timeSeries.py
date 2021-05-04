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


def filter_data(data, time, lowerTimes, upperTimes):
    '''
    Filter out data within a list of certain time ranges
    ------
    parameters:
    data: np.1darray
        array of data to be filtered
    time: np.1darray
        time series of the data taken
    lowerTimes: np.1darray
        array of lower limit of time ranges
    upperTimes: np.ndarray
        arrray of upper limit of time ranges
    ------
    return:
    data_matched: np.1darray
        Data that was filtered out
    idxSecs: np.1darray
        List of indices for different time ranges
    '''
    idxSecs = []
    for i, lowerTime in enumerate(lowerTimes):
        conditions = ((time >= lowerTimes[i]) & (time <= (upperTimes[i])))
        idxSec = np.where(conditions)
        idxSecs.append(idxSec[0])
    idxSecs = np.array(idxSecs)

    data_matched = []
    for idxSec in idxSecs:
        data_temp = data[idxSec]
        data_matched.append(data_temp)

    return data_matched, idxSecs

