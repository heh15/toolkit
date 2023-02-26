import numpy as np

def fit_proportional(x, y):
    '''
    Fit the linear relation forced through 0 point
    ------
    Parameters:
    x: np.1darray
        x of the data
    y: np.1darray
        y of the data
    ------
    return:
    a: float
        coefficient between x and y
    '''
    x = x[:,np.newaxis]
    a,_,_,_ = np.linalg.lstsq(x, y)
    return a
