def Table_sort(surveyTable, columns_array):
    '''
    surveyTable is an astroTable object.
    columns_array is a 2D array for setting criterions for multiple columns
       'and' condition in first dimension. 
       'or' condition in second dimension. 
    '''
    surveyTable_group = surveyTable.group_by('ALMA sanitized source name')
    selectedTable = QTable()
    for source in surveyTable_group.groups.keys['ALMA sanitized source name']: 
        mask = surveyTable_group.groups.keys['ALMA sanitized source name'] == source
        sourceTable = surveyTable_group.groups[mask]
        conditions = []
        for columns in columns_array:
            conditions_temp = []
            for column in columns:
                condition = (True in sourceTable[column])
                conditions_temp.append(condition)
            conditions.append((any(conditions_temp)))
        if all(conditions):
            if len(selectedTable) == 0:
                selectedTable = sourceTable.copy()
            else:
                selectedTable = vstack([selectedTable, sourceTable])

    return selectedTable


def Coordinate_match(df1, df2, columns):

    '''    
    match the coordinates in df1 and df2 and select the certain column in df2 to df1. 
    '''
    df3 = pd.DataFrame(columns = columns, index = df1.index)
    columns_overlap = np.intersect1d(columns, df2.columns)
    columns_exclude = [column for column in columns if column not in columns_overlap]
    for i in df1.index:
        Coord = SkyCoord(df1.loc[i, 'RA']*u.degree, df1.loc[i, 'Dec']*u.degree)
        index = np.where(SkyCoord(df2['RA']*u.degree, df2['Dec']*u.degree).separation(Coord) < 40*u.arcsec)[0]
        if len(index) >0:
            if len(columns_overlap) > 0:
                df3.loc[i, columns_overlap] = df2.loc[index[0], columns_overlap]            
            if len(columns_exclude) > 0:
                df3.loc[i, columns_exclude] = True
        else:
            if len(columns_exclude) > 0:
                df3.loc[i, columns_exclude] = False
                
    df1 = pd.concat([df1, df3], axis=1)
        
    return df1
