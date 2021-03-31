############################################################
# function

def beam_get(imagename, region_init, ratio=2.0, **kwarg):
    ''' 
    parameters
    imagename: The path to the CASA image file
    regions: The path to the CASA region file or the CASA 
    region string. 
    ratio: The ratio to be multiplied by the fitted ellipse to 
    get the final elliptical aperture shape.
    **kwarg: other parameters that goes into the imfit 
    '''
    beam=imfit(imagename=imagename,region=region_init, **kwarg)
    regions = []
    for i in range(beam['results']['nelements']):
        component = 'component' + str(i)
        x_value=beam['results'][component]['shape']\
                 ['direction']['m0']['value']
        y_value=beam['results'][component]['shape']\
                 ['direction']['m1']['value']
        bmaj_value=beam['results'][component]\
                    ['shape']['majoraxis']['value']
        bmin_value=beam['results'][component]['shape']\
                    ['minoraxis']['value']
        pa_value=beam['results'][component]['shape']\
                  ['positionangle']['value']
        x=str(x_value)+'rad'
        y=str(y_value)+'rad'
        bmaj=str(bmaj_value/2.0*ratio)+'arcsec'
        bmin=str(bmin_value/2.0*ratio)+'arcsec'
        pa=str(pa_value)+'deg'
        region='ellipse[['+x+','+y+'],['+bmaj+','+bmin+'],'+pa+']'
        regions.append(region)

    return regions

# need to execute tb.open(vis) first
# need to import the regrid_time() from miscellenious.py
def average_spws(spws, iant=0, spw_template=None):
    '''
    Average data among different spectral windows
    ------
    Parameters:
    spws: list
        List of spectral windows
    iant: int
        The ID of the antennae
    spw_template: float
        One of the spectral windows all the other will be regridded to
    ------
    Return:
    data_avg: np.ndarray
        data that averaged among different spws
    time: np.ndarray
        time corresonding to averaged data 
    '''
    # initalize variables
    if spw_template == None:
        spw_template = spws[0]
    Table = {}

    for i in spws:
        data_label = 'spw '+str(i)+' data'
        time_label = 'spw '+str(i)+' time'
        dat = tb.query('ANTENNA1==%d && ANTENNA2==%d && DATA_DESC_ID ==%d && SCAN_NUMBER not in %s'%(iant,iant,i, scan_ATM))
        data = np.mean(np.real(dat.getcol('DATA')), axis=(0,1))
        time = dat.getcol('TIME')
        Table[data_label] = data
        Table[time_label] = time

    # regrid the data to the time of the first data
    data_columns = []
    for i in spws:
        data_label = 'spw '+str(i)+' data'
        time_label = 'spw '+str(i)+' time'
        regrid_label = 'spw '+str(i)+' regridded'
        data_template = 'spw '+str(spw_template)+' data'
        time_template = 'spw '+str(spw_template)+' time'
        Table[regrid_label] = regrid_time(Table[data_template], Table[data_label],
                                          Table[time_template], Table[time_label])
        data_columns.append(regrid_label)

    # average the data in different spectral windows
    time = np.array(Table[time_template])
    data_avg = np.full((len(data_columns), len(time)), np.nan)
    for i, column in enumerate(data_columns):
        data_avg[i] = Table[column]
    data_avg = np.mean(data_avg, axis=0)

    return data_avg, time

