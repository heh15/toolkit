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
