import numpy as np
import os,glob
import scipy.ndimage as sni
import sys
import re
import itertools
from shutil import copytree
from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import make_lupton_rgb
import aplpy
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib.patches import Arrow
from photutils import SkyCircularAnnulus
from photutils import SkyCircularAperture
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import SkyEllipticalAperture
from photutils import SkyEllipticalAnnulus
from photutils import EllipticalAperture
from photutils import EllipticalAnnulus
from photutils import aperture_photometry
import numpy.ma as ma
import math
import shutil
import pandas as pd

############################################################
# function

# import and cut the file
# Nov 16th,2018. If the data is not in the first item of the header, try importfits and exportfits to convert image. 
def fits_import(fitsimage, item=0):
    hdr = fits.open(fitsimage)[item].header
    wcs = WCS(hdr).celestial
    data=fits.open(fitsimage)[item].data
    data=np.squeeze(data)
    data_masked=np.ma.masked_invalid(data)

    return wcs, data_masked

# example position and size format. 
def cut_2d(data_masked,position,size,wcs):
    cut=Cutout2D(data=data_masked,position=position,size=size,wcs=wcs)
    data_cut=cut.data
    wcs_cut=cut.wcs

    return wcs_cut, data_cut

# draw rings of apertures with give list of radiuses.
def aperture_rings(a_in,a_out,wcs,cosi,pa):
    ring_sky=SkyEllipticalAnnulus(positions=center,a_in=a_in*u.arcsec,a_out=a_out*u.arcsec,b_out=a_out*cosi*u.arcsec,theta=pa*u.degree)
    ring_pix=ring_sky.to_pixel(wcs=wcs)
    return ring_pix

for i in range(1,size): # for multiple rings. 
    rings[i]=aperture_rings(radius_arc[i-1],radius_arc[i],wcs,cosis[i],pas[i])
    rings_mask[i]=Apmask_convert(rings[i],mom2_model)

# convert aperture to the mask
def Apmask_convert(aperture,data_cut):
    apmask=aperture.to_mask(method='center')
    shape=data_cut.shape
    mask=apmask.to_image(shape=((shape[0],shape[1])))
    ap_mask=mask==0
    ap_masked=np.ma.masked_where(ap_mask,data_cut)

    return ap_masked

# convert region to the mask. 
def Regmask_convert(aperture,data_cut):
    apmask=aperture.to_mask()
    shape=data_cut.shape
    mask=apmask.to_image(shape=((shape[0],shape[1])))
    ap_mask=mask==0
    ap_masked=np.ma.masked_where(ap_mask,data_cut)

    return ap_masked

# convert input .mask file to the python mask
def masked_convert(data_masked,region_masked):
    data_mask=data_masked.mask
    region_mask=np.ma.make_mask(region_masked==0)
    region_mask=np.ma.mask_or(data_mask,region_mask)
    data_region=np.ma.masked_where(region_mask,data_masked)
    return data_region

# cut the imported data cube.
def cut_3d(data,position,size,wcs):
    for i in range(data_3d.shape[0]):
        cut=Cutout2D(data=data[i],position=position,size=size,wcs=wcs)
        if i==0:
            data_cut=cut.data
        elif i==1:
            data_cut=np.stack((data_cut,cut.data))
        else:
            temp=np.expand_dims(cut.data,axis=0)
            data_cut=np.concatenate((data_cut,temp))
    wcs_cut=cut.wcs
    return data_cut, wcs_cut

# flux of the region given by mask for radio image
def flux_mask_get(data_region,rms,chans,chan_width):
    flux=np.ma.sum(data_region)/beam_area_pix
    chans_tmp=chans+np.zeros((np.shape(data_region)[0],np.shape(data_region)[1]))
    error=np.sqrt(chans_tmp)*rms*chan_width/sqrt(beam_area_pix)
    error_masked=np.ma.masked_where(data_region.mask,error)
    uncertainty=math.sqrt(np.ma.sum(np.power(error_masked,2)))
    return flux, uncertainty

# flux of the region given by aperture for radio image
def flux_aperture_get(data_masked,aperture,rms,chans,chan_width, beamarea_pix):
    data_cut=data_masked.data
    mask=data_masked.mask
    if np.shape(chans) == ():
        chans = np.full(data_cut.shape, chans) 
    flux=aperture_photometry(data_cut,apertures=aperture,mask=mask)['aperture_sum'][0]/beamarea_pix
    error=np.sqrt(chans)*rms*chan_width/np.sqrt(beamarea_pix)
    uncertainty=aperture_photometry(data_cut,apertures=aperture,mask=mask,error=error)['aperture_sum_err'][0]

    return flux, uncertainty

# mask 3d cube with a region file in specified channels (June 28th)
def Regmask3d(data,region_pix,lowchan,highchan):
    region_masks=region_pix.to_mask()
    if type(region_masks)==list:
        region_mask=region_masks[0]
    else:
        region_mask=region_masks
    shape=np.shape(data)
    mask=region_mask.to_image(shape=((shape[1],shape[2])))
    mask3d=np.zeros((shape[0],shape[1],shape[2]))
    mask3d[lowchan:highchan]=mask
    maskTF=mask3d==1

    data_masked=np.copy(data)
    data_masked[maskTF]='nan'

    return data_masked, maskTF

## bin the image in certain wcs frame. 
def reproj_binning(data, wcs, bin_num):
    
    map_in_shape=np.shape(data)
    nx_in, ny_in=map_in_shape
    nx_out=math.trunc(nx_in/bin_num);ny_out=math.trunc(ny_in/bin_num)
    xs,ys=np.meshgrid(np.arange(nx_out), np.arange(ny_out))
    wcs_out=wcs.deepcopy()
    wcs_out.wcs.crpix =[math.trunc(nx_out/2), math.trunc(ny_out/2)]
    wcs_out.wcs.cdelt=wcs.wcs.cdelt*bin_num
    wcs_out.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    coords_out=pixel_to_skycoord(xs, ys, wcs_out)
    coords_out_flat=coords_out.flatten()
    pixel_labels_out = np.arange(xs.size)
    data_binned=np.zeros((nx_out, ny_out)).flatten()
    map_out_shape=(nx_out, ny_out)
    
    xs_in, ys_in = np.meshgrid(np.arange(nx_in), np.arange(ny_in))
    coords_in = pixel_to_skycoord(xs_in, ys_in, wcs)
    pixel_map_arr = np.full((nx_in, ny_in), np.nan).flatten()

    i_in=0
    npix_in = coords_in.flatten().size
    dra, ddec = np.zeros(npix_in), np.zeros(npix_in)
    i_out, d2d, d3d = match_coordinates_sky(coords_in.flatten(), coords_out_flat)
    dra, ddec = (coords_in.flatten()).spherical_offsets_to(
        coords_out_flat[i_out])
    dra = dra.arcsec
    ddec = ddec.arcsec

    good = (-0.5001 <= dra) & (dra < 0.5001) & (-0.5001 <= ddec) & (ddec < 0.5001)
    pixel_map_arr[good]=pixel_labels_out[i_out[good]]
    data_labeled=np.stack((data.flatten(),pixel_map_arr), axis=1)
    nan_index=np.where(np.isnan(data_labeled[:,1]))
    data_labeled=np.delete(data_labeled, nan_index,axis=0)
    data_labeled=data_labeled[np.argsort(data_labeled[:,1])]
    data_group=np.split(data_labeled[:,0], np.cumsum(np.unique(data_labeled[:,1], return_counts=True)[1])[:-1])
    for i in pixel_labels_out:
        data_binned[i]=np.nanmean(data_group[i])

    data_binned=data_binned.reshape(map_out_shape)
        
    return wcs_out, data_binned


def reproj_binning2(data, wcs_in, bin_num, centerCoord = '', shape_out=''):
    '''
    centerCoord is the array of central coordinates in degree. 
    '''
    map_in_shape=np.shape(data)
    nx_in, ny_in=map_in_shape
    nx_out = math.trunc(nx_in/bin_num); ny_out=math.trunc(ny_in/bin_num)
    if shape_out == '':
        shape_out = (nx_out, ny_out)
    if centerCoord == '':
        centerCoord = wcs_in.wcs.crval
    wcs_out = WCS(naxis = 2)
    wcs_out.wcs.crval = centerCoord
    wcs_out.wcs.crpix =[math.trunc(shape_out[1]/2), math.trunc(shape_out[0]/2)]
    wcs_out.wcs.cdelt=wcs_in.wcs.cdelt*bin_num
    wcs_out.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    data_binned, footprint = reproject_interp((data, wcs_in), wcs_out, shape_out=shape_out)

    
    return wcs_out, data_binned
