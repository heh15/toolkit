from reproject import reproject_interp
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import match_coordinates_sky

def read_ascii(filename, columns, colnumber, skiprows=0):
    '''
    Read the ascii table with fixed column width
    Can be replaced by the pd.readfwf()
    ------
    parameters
    filename: str
        The path to the file with table
    columns: list
        list of column names to be assigned. The name of coordinates 
    are ['RAh', 'RAm', 'RAs', 'Ded', 'Dem', 'Des']
    colnumber: 2D list
        list of start and end column numbers for each column. The start value is 
    the start column number in text file minus one. The end value is the end column
    number. 
    ------
    return
    dataFrame: pd.DataFrame
        DataFrame of the extracted columns
    '''
    dataFrame = pd.DataFrame(columns=columns)
    with open (filename, 'r') as infile:
        text = infile.readlines()
        text = text[skiprows:]
        line = text[0]
        for line in text:
            row = dict.fromkeys(columns)
            for i, column in enumerate(columns):
                lower = colnumber[i][0]
                upper = colnumber[i][-1]
                row[column] = line[lower:upper]
            dataFrame = dataFrame.append(row, ignore_index=True)

    coordsName = ['RAh', 'RAm', 'RAs','Ded', 'Dem', 'Des']
    for column in coordsName:
        if column in columns:
            dataFrame[column] = dataFrame[column].astype(float)

    return dataFrame

def Coords_cal(dataFrame):
    '''
    Calculate the coordinates for the pd.DataFrame extracted from ascii table
    ------
    parameters
    dataFrame: pd.DataFrame
        dataFrame from "read_ascii()" function
    ------
    return
    Table: pd.DataFrame
        Dataframe of coordinates 
    '''
    rah = dataFrame['RAh']; ram = dataFrame['RAm']; ras = dataFrame['RAs']
    ded = dataFrame['Ded']; dem= dataFrame['Dem']; des = dataFrame['Des']
    sign = dataFrame['sign']
    dataFrame['RA'] = 15*(rah+ram/60+ras/3600)
    dataFrame['Dec'] = ded+dem/60+des/3600
    dataFrame['Dec'].loc[dataFrame['sign'] == '-'] = -1*dataFrame['Dec'].loc[dataFrame['sign'] == '-']
    dataFrame = dataFrame.drop('sign', axis=1)

    return dataFrame

def Coords_str2deg(RA, Dec):
    '''
    Convert the coordinates from strings of 'hh:mm:ss' and
    'dd:mm:ss' to value of degrees
    ------
    Parameters:
    RA: pd.Series
        Series of strings for RA coordinates
    Dec: pd.Series
        Series of strings for declination 
    ------
    Return
    RA_deg, Dec_deg: pd.Series
        Series of RA and Dec values in degree. 
    '''
    RA_split = RA.str.split(pat=':', expand=True)
    Dec_split = Decls.str.split(pat=':', expand=True)
    for i in range(3):
        RA_split[i] = RA_split[i].astype(float)
        Dec_split[i] = Dec_split[i].astype(float)
    RA_deg = 15*(RA_split[0]+RA_split[1]/60+RA_split[2]/3600)
    Dec_deg = Dec_split[0]/np.abs(Dec_split[0])\
                * (np.abs(Dec_split[0])+Dec_split[1]/60\
                        +Dec_split[2]/3600)

    return RA_deg, Dec_deg

def match_coords_cart(coords1, coords2):
    '''
    Select the coordinates in df2 that is closest to every object in df1
    ------
    Paramters:
    coords1: np.ndarray
        Numpy array of 3D cartisan coordinates with shape of (:, 3).
    coords2: np.ndarray
        Numpy array of 3D cartisan coordinates with shape of (:, 3). 
    ------
    return
    coords2_matched: np.ndarray
        Indexes of matched object in coords2
    '''
    # get the difference between each vector from coords1 and coords2
    diff = coords1[:,np.newaxis, :] - coords2
    dist = np.sqrt(np.sum(np.square(diff), axis=2))
    coords2_matched = np.nanargmin(dist, axis=1)

    return coords2_matched

def match_coords_skycart(coords1, coords2):
    '''
    Similar to "match_coords_cart" but using Skycoord to match
    ------
    Parameters:
    coords1: Skycoord
        List of 3d cartesian Skycoord coordinates to be matched. Can be created using following
        commands. 
        >>>coords1 = SkyCoord(x=x, y=y, z=z, representation_type='cartesian')
    coords2: Skycoord 
        List of 3d carteisan Skyccord to match coords1
    ------
    Return:
    coords2_matched: np.ndarray
        Array of coords2 that matched to coords1. 
    '''
    indexes, offsets_2d, offsets_3d = coords1.match_to_catalog_3d(coords2)
    coords2_matched = coords2[indexes]

    return coords_matched

def Coordinate_match_fixedRadius(df1, df2, columns, radius=40, RA1col='RA', Dec1col='Dec',
                    RA2col='RA', Dec2col='Dec', mode='closest'):

    '''
    Match the coordinates in df1 and df2 and select the certain column in df2 to df1. An example
    is to match the two catalogus from ALMA archival search.
    ------
    Parameters
    df1: pandas DataFrame
        Catalog of sources to be matched
    df2: pandas DataFrame
        Second catalog of sources to match the first catalog
    columns: list
        List of column names to be extracted from second catalog
    radius: float
        Radius of the circle within which sources are considered
        to be matched. The value is in arcsec.
    RA1col, Dec1col: str
        Name of right asscention and declination column for the first
        catalog
    RA2col, Dec2col: str
        Name of right asscention and declination column for the second
        catalog
    mode: str
        Method to pick the value  if more than one item in second catalog
        are matched. The methods are:
        "closest": select the object with closest offset
        "first": select the object that first appears in the second catalog.
    '''
    df3 = pd.DataFrame(columns = columns, index = df1.index)
    for i in df1.index:
        Coord1 = SkyCoord(df1.loc[i, RA1col]*u.degree, df1.loc[i, Dec1col]*u.degree)
        Coords2 = SkyCoord(df2[RA2col]*u.degree, df2[Dec2col]*u.degree)
        indexes = np.where(Coords2.separation(Coord1) < radius*u.arcsec)[0]
        if len(indexes) >0:
            if mode == 'closest':
                index = np.argmin(Coords2.separation(Coord1))
                df3.loc[i] = df2.loc[index, columns]
            elif mode == 'first':
                df3.loc[i] = df2.loc[indexes[0], columns]
    df1_matched = pd.concat([df1, df3], axis=1)

    return df1_matched

def Coordinate_match_closest(df1, df2, coords1, coords2, columns, newColumns=[], full=False):
    '''
    Match the closest coordinate from coords2 to that of coords1 for the closest target. An example
    of this function is to match the star cluster with its closest GMCs. 
    ------
    Parameters:
    df1: pd.DataFrame
        Pandas data frame that contains information for sources to be matched. Note the index of 
        the first data frame should be reset with "reset_index()" function to start from 0
    df2: pd.DataFrame
        Pandas data frame that contains information to match the first catalog
    coords1: photutils.SkyCoord
        Sky coordinates for first catalog
    coords2: photutils.SkyCoord
        Sky coordinates for second catalog
    columns: list
        List of columns that are extracted from the second catalog
    newColumns: list
        List of column names that contains information extracted from second 
        catalog. 
    full: bool
        If false, only returns two extracted dataframes. If false, also return
        output from 'SkyCoord.match_to_catalog_sky()' function. 
    ------
    Return:
    df1_matched:
        Data frame that contains first catalog and matched information from 
        second catalog. 
    df2_matched:
        Data frame that extracted from the second catalog to match the first 
        catalog. 
    idx, d2d, d3d: np.array
        Output from SkyCoord.match_to_catalog_sky() function respresenting indexes,
        2d offsets and 3d offsets. 
    '''
    idx, d2d, d3d = coords1.match_to_catalog_sky(coords2)
    df2_toMatch = df2.loc[idx].reset_index()
    df1_matched = df1.copy(deep=True)
    if len(newColumns)== 0:
        newColumns = columns
    for i, newColumn in enumerate(newColumns):
        df1_matched[newColumn] = df2_toMatch[columns[i]]    
        
    if full == False:
        return df1_matched, df2_toMatch
    else:
        return df1_matched, df2_toMatch, idx, d2d, d3d

def group_catalog_to_pix(coords_in,wcs,data):
    '''
    Give the label for a catalog of objects that belongs to each pixel
    ------
    Parameters:
    coords_in: SkyCoord object
        List of sky coordinates.
    wcs: WCS
        WCS object of the data
    data: np.2darray
        Numpy array of data.
    ------
    Return:
    matched_idx: np.mask
        Numpy mask array to determine if the object belongs to a certain pixel
    coords_label: np.1darray
        Numpy array of labels coorresponding to the pixel number for each object in
        the catalog. 
    '''
    deltax, deltay = np.abs(wcs.wcs.cdelt) * 3600
    ny, nx = np.shape(data)
    xs, ys = np.meshgrid(np.arange(nx), np.arange(ny))
    coords_pix = pixel_to_skycoord(xs, ys, wcs)
    pixel_labels_out = (np.arange(xs.size)).astype(int)
    
    idx, d2d, d3ds = coords_in.match_to_catalog_sky(coords_pix.flatten())
    dra, ddec = coords_in.spherical_offsets_to(
        coords_pix.flatten()[idx])
    dra = dra.arcsec
    ddec = ddec.arcsec
    good = (-deltax/2-0.01 <= dra) & (dra < deltax/2+0.01) & (-deltay/2-0.01 <= ddec) & (ddec < deltay/2+0.01)
    coords_labels = np.full(np.shape(coords_in),np.nan)

    coords_labels[good] = pixel_labels_out[idx[good]]
    matched_idx = good
    
    return matched_idx, coords_labels
