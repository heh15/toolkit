def count_time(stop, start):
    '''
    Convert the time difference into human readable form. 
    '''
    dure=stop-start
    m,s=divmod(dure,60)
    h,m=divmod(m,60)
    print("%d:%02d:%02d" %(h, m, s))

    return

def initialize_value(variable, value):
    '''
    initialize the value of the variable if is not defined before. 
    '''
    try:
        variable
    except:
        variable = value

    return

from math import log10, floor
def round_sig(x, sig=2):
    '''
    round the x to certain significant figures
    '''
    return round(x, sig-int(floor(log10(abs(x))))-1)


def find_nearest(array, value):
    '''
    Find the index of the value in the array closest to the given value
    ------
    Parameters:
    array: numpy.ndarray
        numpy array to be matched
    value: float64
        value to be matched with numpy array
    ------
    Return:
    idx: int
        index of the closest value in array.
    '''
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))

    return idx


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

