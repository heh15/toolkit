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
