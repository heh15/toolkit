import pandas as pd
from astroquery.alma import Alma
from astroquery.ned import Ned
import numpy as np
import time
import re
from astropy.table import Table
import sys, os
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.alma import Alma
from astroquery.ned import Ned
from astropy.table import join, vstack, unique
from astropy.table import Table, QTable
import pickle
from astropy.io import ascii


def match_lines(surveyTable, columns_array):

    surveyTable_group = surveyTable.group_by('NED source name')
    selectedTable = QTable()
    for source in surveyTable_group.groups.keys['NED source name']: 
        mask = surveyTable_group.groups.keys['NED source name'] == source
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

    sourcelist = pd.DataFrame()
    if len(selectedTable) == 0:
        print('No target found')
    else:
        names=selectedTable.group_by('NED source name').groups.keys['NED source name']
        names=list(names)
        RAs=[];Decs=[]
        for source in names:
            mask = surveyTable_group.groups.keys['NED source name'] == source
            sourceTable = surveyTable_group.groups[mask]
            RAs.append(sourceTable['ALMA RA'][0])
            Decs.append(sourceTable['ALMA Dec'][0])

        sourcelist['name']=names; sourcelist['RA']=RAs; sourcelist['Dec']=Decs
                
    return selectedTable, sourcelist

# def match_lines(Results, columns_array):

#     selectedTable = QTable()
#     names = []
#     for source in Results.keys(): 
#         conditions = []
#         for columns in columns_array:
#             conditions_temp = []
#             for column in columns:
#                 condition = (True in Results[source][column])
#                 conditions_temp.append(condition)
#             conditions.append((any(conditions_temp)))
#         if all(conditions):
#             names.append(source)
#             if len(selectedTable) == 0:
#                 selectedTable = Results[source].copy()
#             else:
#                 selectedTable = vstack([selectedTable, Results[source].copy()])
                
#     sourcelist = pd.DataFrame()
#     if len(selectedTable) == 0:
#         print('no targets found')
#     else: 
#         RAs=[];Decs=[]
#         for source in names:
#             RAs.append(Results[source]['ALMA RA'][0])
#             Decs.append(Results[source]['ALMA Dec'][0])

#         sourcelist['name']=names; sourcelist['RA']=RAs; sourcelist['Dec']=Decs
                
#     return selectedTable, sourcelist


def find_NEDCoords(sourceNames, searchradius=1*u.arcmin):
    Coords = []
    for source in sourceNames:
        result_table = Ned.query_object(source)
        RA = result_table['RA'][0]*u.degree
        Dec = result_table['DEC'][0]*u.degree
        Coord = [SkyCoord(ra=RA, dec=Dec), searchradius]
        Coords.append(Coord)

    return Coords


def stack_table(queryResults):
    targets=list(queryResults.keys())
    surveyTable=QTable()
    freqrangelist=[]
    freqreslist=[]
    linesenslist=[]
    linesensnlist=[]
    pollist=[]
    for i in range(len(targets)):
        if len(queryResults[targets[i]])>0:
            tempTable = queryResults[targets[i]].copy()
            for freqra in tempTable['Frequency ranges']:
                freqrangelist.append(freqra)
            # for freqres in tempTable['Frequency resolution']:
            #     freqreslist.append(freqres)
            # for linesens in tempTable['Line sensitivity (10 km/s)']:
            #     linesenslist.append(linesens)
            # for linesensn in tempTable['Line sensitivity (native)']:
            #     linesensnlist.append(linesensn)
            # for pol in tempTable['Pol products']:
            #     pollist.append(pol)

            tempTable.remove_column('Frequency ranges')
            # tempTable.remove_column('Frequency resolution')
            # tempTable.remove_column('Line sensitivity (10 km/s)')
            # tempTable.remove_column('Line sensitivity (native)')
            # tempTable.remove_column('Pol products')
            if len(surveyTable)==0:
                surveyTable=tempTable.copy()
            else:
                surveyTable=vstack([surveyTable, tempTable])
    
    surveyTable['Frequency ranges']=freqrangelist; surveyTable['Frequency ranges'].unit=u.GHz
    # surveyTable['Frequency resolution']=freqreslist; surveyTable['Frequency resolution'].unit=u.kHz
    # surveyTable['Line sensitivity (10 km/s)']=linesenslist; surveyTable['Line sensitivity (10 km/s)'].unit=u.mJy/u.beam
    # surveyTable['Line sensitivity (native)']=linesensnlist; surveyTable['Line sensitivity (native)'].unit=u.mJy/u.beam
    # surveyTable['Pol products']=pollist
    columns=list(surveyTable.columns)
    index=columns.index('NED source name')
    columns.insert(0,columns.pop(index))
    index=columns.index('scientific_category')
    columns.insert(2,columns.pop(index))
    index=columns.index('science_keyword')
    columns.insert(3,columns.pop(index))
    surveyTable=surveyTable[columns]


    return surveyTable




def Coordinate_match(df1, df2, columns):

    '''    
    match the coordinates in df1 and df2 and select the certain column in df2 to df1.
    ---
    Parameters:
    df1: pd.DataFrame
        The pandas dataframe to be matched. The Coordinates column names are 'RA' and 'Dec'. 
    The units are degrees. 
    df2: pd.DataFrame
        The pandas dataframe to match the df1.
    columns: list
        The columns in df2 to be imported to df1. If the column name is not in df2, then it
    exports True or False value in the newly built column.  
    ------
    Return:
    df1: pd.DataFrame
        Returned df1 with matched information from df2 
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


def Ned_query_dist(df, H_0=69.6):
    distances = []
    for index in df.index:
        coordinate = SkyCoord(ra=df['RA'][index]*u.degree, dec= df['Dec'][index]*u.degree)
        result = Ned.query_region(coordinate, radius=20*u.arcsec)
        typeInds = np.where(result['Type'] != b'G')
        result.remove_rows(typeInds)
        if len(result) > 0:
            if np.where(~np.isnan(result['Velocity']))[0].size > 0:
                index2 = np.where(~np.isnan(result['Velocity']))[0]
                velocity = result['Velocity'][index2[0]]                     
                distance = velocity/H_0
                distances.append(distance)
            else:
                distances.append(np.nan)
        else:
            distances.append(np.nan)

    df['distance'] = distances

    return df
