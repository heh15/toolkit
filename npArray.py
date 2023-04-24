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
    Equivalent to `np.where(np.isin(elements, test_elements))[0]
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

def assign_matched_values(array, values, array_id, values_id):
    '''
    Assign the array with an array of values where the corresondning
    identities of the array is equal to the identities of the values. 
    ------
    Parameters
    array: np.ndarray
        Input array to be given values
    values: np.ndarray
        Array of values that is to be given to the input array. 
    array_id: np.ndarray
        Array of identities corresponding to the input array. The 
        shape of `array_id` should be the same as `array`. 
    values_id: np.ndarray
        Array of identties corresonding to the assigned values. The
        shape of `values_id` should be the same as `values`. 
    ------
    Return
    array_out: np.ndarray
        The array with values assigned. 
    '''
    array_out = np.copy(array)
    
    dist = np.abs(values_id[:,np.newaxis] - array_id) 
    idx = np.nanargmin(dist, axis=0)
    diff = np.nanmin(dist, axis=0)

    idx_matched = np.where(diff==0)
    if len(idx_matched) > 0:
        array_out[idx_matched] = values[idx[idx_matched]]  

    return array_out

def map_series_by_dict(a, d):
    '''
    Used for substituting each value in the array with another value
    derived from the dictionary (substitute key with corresonding 
    value. It is the same as `pandas.Series.map(dict)` function. 
    ------
    Parameters
    a: np.ndarray
        Numpy array with values to be substituted
    d: dict
        The dictionary corresponding each element in the array with
        values used to substitute. 
    ------
    out_ar: np.ndarray
        The array with substitute values. 
    '''
    v = np.array(list(d.values()))
    k = np.array(list(d.keys()))
    sidx = k.argsort()
    out_ar = v[sidx[np.searchsorted(k,a,sorter=sidx)]]

    return out_ar

