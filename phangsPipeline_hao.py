'''
June 14th, create this script to modify the phangspipline function.
Hao He 
1) Add the 'field' in the split command input and pass the `gal` into this 
parameter. 
2) Add the `spw` option for the split command. 
3) Add the 'collapse_width' option for the do_collapse step. 
4) comment out reading information from the mosaic file.
5) split instead of copy the original calibrated file into the directory. 
'''
import numpy as np

import line_list
import os
import analysisUtils as au

# impart CASA specific task.
from concat import concat
from exportfits import exportfits
from flagdata import flagdata
from imhead import imhead
from immath import immath
from imstat import imstat
from imregrid import imregrid
from importfits import importfits
from makemask import makemask
from mstransform import mstransform
from split import split
from statwt import statwt
from tclean import tclean
from uvcontsub import uvcontsub
from visstat import visstat

def extract_continuum(
    in_file=None,
    out_file=None,
    lines_to_flag=None,
    gal='',
    spw='',
    vsys=0.0,
    vwidth=500.,
    do_statwt=True,
    do_collapse=True,
    quiet=False, 
    collapse_width=10000):
    """
    Extract a continuum measurement set, flagging any specified lines,
    reweighting using statwt, and then binning the continuum channels. 
    """

    sol_kms = 2.99e5

    # Set up the input file

    if os.path.isdir(in_file) == False:
        if quiet == False:
            print("Input file not found: "+in_file)
        return

    # pull the parameters from the galaxy in the mosaic file

#     if gal != None:
#       mosaic_parms = read_mosaic_key()
#       if mosaic_parms.has_key(gal):
#           vsys = mosaic_parms[gal]['vsys']
#           vwidth = mosaic_parms[gal]['vwidth']
#
    # set the list of lines to flag

    if lines_to_flag == None:
        lines_to_flag = line_list.lines_co + line_list.lines_13co + line_list.lines_c18o + line_list.lines_cn

    # Make a continuum copy of the science data

    os.system('rm -rf '+out_file)
    os.system('rm -rf '+out_file+'.flagversions')

#    command = 'cp -r -H '+in_file+' '+out_file
#    print command
#    var = os.system(command)
#    print var
#
    split(vis=in_file,
          outputvis=out_file,
          datacolumn='DATA',
          keepflags=False, 
	  field=gal,
	  spw=spw)       
    
    os.system('rm -rf '+out_file+'.temp_copy')
    
    command = 'mv '+out_file+' '+out_file+'.temp_copy'
    print command
    var = os.system(command)
    
    mstransform(vis=out_file+'.temp_copy',
		outputvis=out_file,
		outframe='Bary', 
		datacolumn='DATA')  
 
    # Figure out the line channels and flag them

    vm = au.ValueMapping(out_file)

    spw_flagging_string = ''
    first = True
    for spw in vm.spwInfo.keys():
        this_spw_string = str(spw)+':0'
        if first:
            spw_flagging_string += this_spw_string
            first = False
        else:
            spw_flagging_string += ','+this_spw_string

    for line in lines_to_flag:
        rest_linefreq_ghz = line_list.line_list[line]

        shifted_linefreq_hz = rest_linefreq_ghz*(1.-vsys/sol_kms)*1e9
        hi_linefreq_hz = rest_linefreq_ghz*(1.-(vsys-vwidth/2.0)/sol_kms)*1e9
        lo_linefreq_hz = rest_linefreq_ghz*(1.-(vsys+vwidth/2.0)/sol_kms)*1e9

        spw_list = au.getScienceSpwsForFrequency(out_file,
                                                 shifted_linefreq_hz)
        if spw_list == []:
            continue

        print "Found overlap for "+line
        for this_spw in spw_list:
            freq_ra = vm.spwInfo[this_spw]['chanFreqs']
            chan_ra = np.arange(len(freq_ra))
            to_flag = (freq_ra >= lo_linefreq_hz)*(freq_ra <= hi_linefreq_hz)
            to_flag[np.argmin(np.abs(freq_ra - shifted_linefreq_hz))]
            low_chan = np.min(chan_ra[to_flag])
            hi_chan = np.max(chan_ra[to_flag])
            this_spw_string = str(this_spw)+':'+str(low_chan)+'~'+str(hi_chan)
            if first:
                spw_flagging_string += this_spw_string
                first = False
            else:
                spw_flagging_string += ','+this_spw_string

    print "... proposed flagging "+spw_flagging_string

    if spw_flagging_string != '':
        flagdata(vis=out_file,
                 spw=spw_flagging_string,
                 )

    # Here - this comman needs to be examined and refined in CASA
    # 5.6.1 to see if it can be sped up. Right now things are
    # devastatingly slow.
    if do_statwt:
        print "... deriving empirical weights using STATWT."
        statwt(vis=out_file,
               timebin='0.001s',
               slidetimebin=False,
               chanbin='spw',
               statalg='classic',
               datacolumn='DATA',
               )

    if do_collapse:
        print "... Binning the continuum channels"

        os.system('rm -rf '+out_file+'.temp_copy')
        os.system('rm -rf '+out_file+'.temp_copy.flagversions')
                                                                
        command = 'mv '+out_file+' '+out_file+'.temp_copy'
        print command
        var = os.system(command)
        print var

        command = 'mv '+out_file+'.flagversions '+out_file+'.temp_copy.flagversions'
        print command
        var = os.system(command)
        print var

        split(vis=out_file+'.temp_copy',
              outputvis=out_file,
              width=collapse_width,
              datacolumn='DATA',
              keepflags=False)       

        os.system('rm -rf '+out_file+'.temp_copy')
        os.system('rm -rf '+out_file+'.temp_copy.flagversions')
    
    os.system('rm -rf '+out_file+'.temp_copy')
    os.system('rm -rf '+out_file+'.flagversions')
        
    return 
