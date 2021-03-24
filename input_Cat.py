def read_ascii(filename, columns, colnumber):
    '''
    Read the ascii table with fixed column width
    ------
    parameters
    filename: str
        The path to the file with table
    columns: list
        list of column names to be assigned. The name of coordinates 
    are ['RAh', 'RAm', 'RAs', 'Ded', 'Dem', 'Des']
    colnumber: 2D list
        list of start and end column numbers for each column
    ------
    return
    dictionary: dict
        dictionary of extracted columns. 
    ''' 
    Table3 = pd.DataFrame()
    dictionary = dict.fromkeys(columns)
    for i in range(len(columns)):
        dictionary[columns[i]] = {'value': [], 'colnumber': colnumber[i]}
    with open (filename, 'r') as infile:
        text = infile.readlines()
        text = text[152:]
        line = text[0]
        for line in text:
            for key in dictionary.keys():
                lower = dictionary[key]['colnumber'][0]
                upper = dictionary[key]['colnumber'][-1]
                dictionary[key]['value'].append(line[lower:upper])

    coordsName = ['RAh', 'RAm', 'RAs','Ded', 'Dem', 'Des']
    for key in coordsName:
        dictionary[key]['value'] = np.float_(dictionary[key]['value'])

    return dictionary

def Coords_cal(dictionary):
    '''
    Calculate the coordinates for the dictionary extracted from ascii table
    ------
    parameters
    dictionary: dict
        dictionaries from "read_ascii()" function
    ------
    return
    Table: pd.DataFrame
        Dataframe of coordinates 
    '''
    Table = pd.DataFrame()
    rah = dictionary['RAh']['value']; ram = dictionary['RAm']['value']; ras = dictionary['RAs']['value']
    ded = dictionary['Ded']['value']; dem= dictionary['Dem']['value']; des = dictionary['Des']['value']
    sign = dictionary['sign']['value']
    Table['RA'] = 15*(rah+ram/60+ras/3600)
    Table['Dec'] = ded+dem/60+des/3600; Table['sign'] = sign
    Table['Dec'].loc[Table['sign'] == '-'] = -1*Table['Dec'].loc[Table['sign'] == '-']
    Table = Table.drop('sign', axis=1)

    return Table
