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

# round the value to the first digit of the uncertainty
def round_to_error(value, error):
    """
    Round value and error to the first significant digit of the error.

    Example:
        value = 12.3456, error = 0.0789
        -> 12.3, 0.08
    """
    if error == 0 or not np.isfinite(error):
        return value, error

    error_abs = abs(error)
    
    # decimal position of first significant digit of error
    exponent = np.floor(np.log10(abs(error)))
    decimals_err = int(-exponent)
    error_rounded = np.round(error, decimals_err)
    
    # Now determine decimal places from the rounded error
    exponent_rounded = np.floor(np.log10(abs(error_rounded)))
    decimals_value = int(max(0, -exponent_rounded))

    value_rounded = np.round(value, decimals_value)
    error_rounded = np.round(error_rounded, decimals_value)
    
    return value_rounded, error_rounded

# fit the Gaussian function
def gaus(x,a,x0,sigma):
    '''
    Fit the gaussian spectrum
    ------
    Parameters
    x: 1D array
    Data of x axis for the Gaussian fitting
    a, x0, sigma: float
    Parameters for th function. 
    ------
    Examples:
    from scipy.optimize import curve_fit
    curve_fit(gaus, x, y, p0=[maximum, mean, sigma])
    sns. 
    '''
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
