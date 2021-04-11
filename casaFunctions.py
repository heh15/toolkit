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

def average_Tsys(Tsys_spectrum, chan_trim=5):
    '''
    Average the system temperature over the frequency axis
    ------
    Parameters
    Tsys_spectrum: np.ndarray
        The extracted Tsys spectrum from tb.getcol()
    chan_trim: int
        number of channels trimed at the edge of spectrum
    ------
    Return 
    Tsys: np.ndarray
        Averaged Tsys
    '''
    Tsys_avg1 = np.mean(Tsys_spectrum, axis=0)
    Tsys_avg2 = Tsys_avg1[chan_trim: (len(Tsys_avg1)-chan_trim)]
    Tsys = np.mean(Tsys_avg2, axis=0)

    return Tsys

def select_spw_Tsys(Tsys, spw):
    '''
    Select Tsys with given spectral window
    ------
    Parameters
    Tsys: np.ndarray
        Averaged Tsys
    spw: int
        Spectral window to be selected
    ------
    Return
    Tsys_sinspw: np.ndarray
        Tsys with single spectral window
    '''
    Tsys_temp = np.copy(Tsys)
    Tsys_temp[np.where(spws!=spw_Tsys)] = np.nan
    Tsys_sinspw = Tsys_temp[~np.isnan(Tsys_temp)]

    return Tsys_sinspw

def normalize_Tsys(Tsys_sinspw, isin_phase, isin_sci, isin_bpass):
    '''
    Normalize Tsys by the start of Tsys for phasecal, science and bandpass 
    respectively
    ---
    Parameters
    Tsys_sinspw: np.ndarray
        Tsys for single spectral window
    isin_phase: np.ndarray
        Indexes of Tsys for phasecal
    isin_sci: np.ndarray
        Indexes of Tsys for science observation
    isin_bpass: np.ndarray
        Indexes of Tsys for bandpass
    ------
    Return
    Tsys_norm: np.ndarray
        Normalized Tsys
    '''
    Tsys_norm = np.full(np.shape(Tsys_sinspw), np.nan)
    Tsys_norm[isin_phase] = Tsys_sinspw[isin_phase] / 
        Tsys_sinspw[isin_phase[0][0]]
    Tsys_norm[isin_sci] = Tsys_sinspw[isin_sci] / 
        Tsys_sinspw[isin_sci[0][0]]
    Tsys_norm[isin_bpass] = Tsys_sinspw[isin_bpass] / 
        Tsys_sinspw[isin_sci[0][0]]

    return Tsys_norm

