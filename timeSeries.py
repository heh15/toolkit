def regrid_time(xdata, ydata, xtime, ytime,
                interval=1.0):
    '''
    ------
    Regrid the data from one time series to another. Not the that 
    the orignal and targeted time series should not contain nan value.  
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


def rebin_time(data, time, timebin=10, method='mean'):
    '''
    ------
    Rebin the data by a certain interval of time. Note the input time
    series should not contain nan value 
    paramters:
    data: np.1darray.
        array of data to be binned
    time: np.1darray
        array of time the data is taken
    timebin: float
        The interval of time to bin the data.
    method: str
        Options for all the data values are binned. The options are:
        'mean'
            Calculate the mean of the data at each binned interval
        'counts"
            Get the data that happens most frequently at each binned
            interval. 
    ------
    retrun: 
    data_binned: np.1darray
        The rebinned data. 
    time_sep: np.1darray
        Rebinned time with interval equal to the timebin. 
    '''
    time_sep = np.arange(time[0], time[-1], timebin) + timebin/2
    data_binned = np.full(np.shape(time_sep), np.nan)

    # following code are similar as that in regrid_time()
    dist = np.abs(time[:, np.newaxis] - time_sep)
    potentialClosest = np.nanargmin(dist, axis=1)
    diff = np.nanmin(dist, axis=1)
    # leave out the time with spacing greater than the interval of original time.
    data[np.where(diff > timebin)] = np.nan
    closestFound, closestCounts = np.unique(potentialClosest, return_counts=True)
    data_group = np.split(data, np.cumsum(closestCounts)[:-1])
    if method == 'mean':
        for i, index in enumerate(closestFound):
            data_binned[index] = np.nanmean(data_group[i])
    if method = 'counts':
        for i, index in enumerate(closestFound):
            value, counts = np.unique(data_group[i], return_counts=True)
            ind = np.nanargmax(counts)
            data_binned[index] = value[ind]

    return data_binned, time_sep


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

