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

# need to import the regrid_time() from miscellenious.py
def average_spws(vis, spws, iant=0, spw_template=None):
    '''
    Average data among different spectral windows
    ------
    Parameters:
    vis: str
        Path to the measurement set 
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
    # read table 
    tb.open(vis)

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

    tb.close()
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


def average_Tsys(Tsys_spectrum, chan_trim=5, average_spw=False, spws=[]):
    '''
    Average the system temperature over the frequency axis
    ------
    Parameters
    Tsys_spectrum: np.ndarray
        The extracted Tsys spectrum from tb.getcol()
    chan_trim: int
        number of channels trimed at the edge of spectrum
    average_spw: bool
        This determines whether to average Tsys measurements over different 
    spectral windows. 
    spws: np.ndarray
        The extracted spectral window array for Tsys measurements
    ------
    Return 
    Tsys: np.ndarray
        Averaged Tsys
    '''
    Tsys_avg1 = np.mean(Tsys_spectrum, axis=0)
    Tsys_avg2 = Tsys_avg1[chan_trim: (len(Tsys_avg1)-chan_trim)]
    Tsys = np.mean(Tsys_avg2, axis=0)

    if (average_spw == True) and len(spws) != 0:
        spw_unique = np.unique(spws)
        shape = (len(spw_unique), len(Tsys)/len(spw_unique))
        Tsys_temp = np.full(shape, np.nan)
        for i, spw in enumerate(np.unique(spws)):
            Tsys_temp[i] = Tsys[np.where(spws==spw)]

        Tsys = np.mean(Tsys_temp, axis=0)

    return Tsys

def select_spw_Tsys(Tsys, spws, spwid):
    '''
    Select Tsys with given spectral window
    ------
    Parameters
    Tsys: np.ndarray
        Averaged Tsys
    spws: np.ndarray
        Array of spectral window ids attached to each measurement
    spwid: int
        Spectral window to be selected
    ------
    Return
    Tsys_sinspw: np.ndarray
        Tsys with single spectral window
    '''

    Tsys_sinspw = Tsys[np.where(spws==spwid)]

    return Tsys_sinspw


def normalize_Tsys(Tsys_sinspw, isin_phase, isin_sci, isin_bpass, normScans=[0,0,0]):
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
    normScans: list
        List of indices of Tsys to be normalized for each type of data
    ------
    Return
    Tsys_norm: np.ndarray
        Normalized Tsys
    '''
    Tsys_norm = np.full(np.shape(Tsys_sinspw), np.nan)
    isins = [isin_phase, isin_sci, isin_bpass]

    for i, isin in enumerate(isins):
        if len(isin[0]) == 0:
            continue
        else:
            Tsys_norm[isin] = Tsys_sinspw[isin] / Tsys_sinspw[isin[0][normScans[i]]]

    return Tsys_norm

def approxCalcPWV(tsrc0,tsrc1,tsrc2,tsrc3,m_el,Tamb=275,r=[1.193,0.399,0.176,0.115],
               tau0=[0.027,0.02,0.01,0.01],eta_c=0.97,verb=True):
    '''
    Calculate the pwv values from the Twvr from 4 channels
    ------
    Parameters
    tsrc0, tsrc1, tsrc2, tsrc3:
        WVR temperature for channel 0, 1, 2, 3
    m_el: float
        Elevation in degrees
    Tamb: float
        Ambient temperature in Kelvin
    r: list
        ratio of opacity at each filter compared to PWV
    tau0: list
        contribution of the continuum
    eta_c: float
        The forward efficiency of the antenna
    verb: bool
        If true, print out bunch of information
    ------
    Return
    pwv_z:
        Zenith pwv values
    '''
    m_el=m_el/57.295   # convert to radians
    T_loss=275.0

    if tsrc0 > T_loss: T_loss = tsrc0

    tsrc0=(tsrc0-(1.0-eta_c)*T_loss)/eta_c
    tsrc1=(tsrc1-(1.0-eta_c)*T_loss)/eta_c
    tsrc2=(tsrc2-(1.0-eta_c)*T_loss)/eta_c
    tsrc3=(tsrc3-(1.0-eta_c)*T_loss)/eta_c

    pw=[0.0,0.0,0.0,0.0]; pw_noc=[0.0,0.0,0.0,0.0]
    site="AOS"

    if site=="AOS":
    # approximate physical temp of atmosphere ,based on ambient temperature, Tamb in Kelvin.
       Tphys=Tamb

    if tsrc0 > Tphys: Tphys = tsrc0
    if verb: print(site, Tphys)

    tel=[0.0,0.0,0.0,0.0]
    tz=[0.0,0.0,0.0,0.0]
    wt=[0.0,0.0,0.0,0.0]

    # calculates transmissions:
    tel[3]=(1.0-tsrc3/Tphys)
    tel[2]=(1.0-tsrc2/Tphys)
    tel[1]=(1.0-tsrc1/Tphys)
    tel[0]=(1.0-tsrc0/Tphys)

    if verb: print("Ta:",tsrc0,tsrc1,tsrc2,tsrc3)
    if verb: print("tel", tel)
    for i in range(4):
        if tel[i]<0.0001: tel[i]=0.0001
        wt[i]=1.0-(abs(tel[i]-0.5)/0.5)**0.5  # weights
    if verb: print('weights ',wt)
    use=1
    for i in range(4):
           pw[i]=-(np.log(tel[i])+tau0[i])/r[i]
    if verb: print('uncorrected pwv',pw)

# wet components
    rat31_1=pw[3]/pw[1]
    pwm=np.mean(pw)
    if pwm>0.5:   # only look for a wet cloud component if pwv>0.5mm   (bit arbitrary cutoff but probably ok)
        pwt=np.zeros(4)
        tauc0_0=-0.02
        tauc0=0.01
        i=0
        std_pwt_old=9999.0
        iloop=True
        std_pwt=np.zeros(25);tauc=np.zeros(25)
        while(iloop):
            tauc[i]=tauc0_0 + tauc0*1.17**i-tauc0
            for i1 in range(4):
                pwt[i1]=-(np.log(tel[i1])+tau0[i1]+tauc[i])/r[i1]
                if pwt[i1]<0.0: iloop=False
            mean_pwt,std_pwt[i]=weighted_avg_and_std(pwt, wt)   # get slope of 4 pwv values, using weights (should weight down channels 0,1 a lot)

            if abs(std_pwt[i])>std_pwt_old or (abs(std_pwt[i])/mean_pwt < 0.000001) : iloop=False # stop loop if slope is getting larger, or diff is <0.001%
            if verb: print('tauc',tauc[i],'std(pwv)',std_pwt[i],'pwv',pwt)
            std_pwt_old=std_pwt[i]
            i+=1
            if i>24: iloop=False


        tau_constant=tauc[i-2]
#   print 'tauc:',tau_constant    # this is the last but one estimate, before it started increasing again
    else:
        tau_constant=0.0  # default, for low pwv

#   re-estimates pwv, after removing additional tau_constant component
    for i in range(4):
           pw_noc[i]=-(np.log(tel[i])+tau0[i]+tau_constant)/r[i]
    if verb: print('corrected:',pw_noc)
    rat31_2=pw[3]/pw[1]
    ratio31_a=pw[3]/pw[1]

#   estimates weighted mean pwv, with and without cloud component:
    ws=0.0
    for i in range(4):
          ws=ws+pw[i]*wt[i]
    pwv_los=ws/sum(wt)
    pwv_z=pwv_los*math.sin(m_el)

#   wet components
#   now optionally remove wet cloud component
    ws=0.0
    for i in range(4):
          ws=ws+pw_noc[i]*wt[i]
    pwv_los_noc=ws/sum(wt)
    pwv_z_noc=pwv_los_noc*math.sin(m_el)
    tau_constant_z=tau_constant      # *math.sin(m_el)   ##!! assume tau_constant is not planar - just the line of sign value (ie don't use sin(elev) )

    return pwv_z
