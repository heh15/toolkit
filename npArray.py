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



