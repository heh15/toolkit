def find_nearest(array, values, full=False):
    '''
    Find the index of the values in the array closest to the given value
    ------
    Parameters:
    array: numpy.ndarray
        numpy array to be matched
    values: numpy.ndarray or float
        values to be matched with numpy array
    full: bool, optional
        Switch determining the nature of return values. When it is False
        (default) just the index returned. When True, the difference 
        between matched array and given values are also returned
    ------
    Return:
    idx: int or int numpy array
        index of the closest value in array.
    diff: float or float numpy array
        Present only if `full`=True. Absolute difference between the 
        given values and matched values in the array. 
    '''
    array = np.asarray(array)
    dist = np.abs(array[:,np.newaxis] - values)
    idx = np.nanargmin(dist, axis=0)
    diff = np.nanmin(dist, axis=0)

    if full:
        return idx, diff
    else:
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



