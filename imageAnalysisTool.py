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
from astropy.coordinates import match_coordinates_sky

###########################################################
# I/O of data

def fits_import(fitsimage, item=0):
    '''
    Import the fits image data.
    ------
    Parameters:
    fitsimage: string
        Path to the fits image.
    item:
        index number in fits hdu file that contains actual data
    '''
    hdr = fits.open(fitsimage)[item].header
    wcs = WCS(hdr).celestial
    data=fits.open(fitsimage)[item].data
    data=np.squeeze(data)

    return wcs, data

def output_fits(fitsimage, data, wcs):
    '''
    Parameters
    ----------
    fitsimage : str
        Filename of the fits image
    data : np.2darray
        Data of the image
    wcs : wcs
        wcs information

    Returns
    -------
    None.

    '''
    header = wcs.to_header()
    hdu = fits.PrimaryHDU(data, header)
    hdu.writeto(fitsimage, overwrite=True)
    
    return

###########################################################
# Select data

def cut_2d(data, position, size, wcs):
    '''
    Cut the image into given rectangle
    ------
    Parameters:
    data: np.2darray
        Image data. 
    position: sky coordinates
        Given by "SkyCoord(ra=ra,dec=dec,frame='fk5'"
    size: 2d quantity
        Given by "u.Quantity((x,y), u.arcsec)"
    wcs: 
        wcs information for the data
    ------
    Return:
    wcs_cut: 
        wcs information for the cut data
    data_cut:
    '''
    cut=Cutout2D(data=data,position=position,size=size,wcs=wcs)
    data_cut=cut.data
    wcs_cut=cut.wcs

    return wcs_cut, data_cut


def aperture_rings(a_in,a_out,wcs,cosi,pa):
    '''
    Generate a list of pixel aperture rings
    '''
    ring_sky=SkyEllipticalAnnulus(positions=center,a_in=a_in*u.arcsec,a_out=a_out*u.arcsec,b_out=a_out*cosi*u.arcsec,theta=pa*u.degree)
    ring_pix=ring_sky.to_pixel(wcs=wcs)
    return ring_pix


def Apmask_convert(aperture_pix,data):
    '''
    Convert photutils.apeture_pix into numpy masked array
    ------
    Parameters:
    aperture_pix: photutils.aperture_pix
        Can be converted from skyaperture using "aperture_sky.to_pixel(wcs)". 
    data: np.2darray
    ------
    Returns:
        data_region: numpy masked array, values in region are masked  
    '''
    apmask=aperture_pix.to_mask(method='center')
    shape=data.shape
    mask=apmask.to_image(shape=((shape[0],shape[1])))
    ap_mask=mask==0
    data_region = np.ma.masked_where(ap_mask,data)

    return data_region

def Regmask_convert(region_pix,data):
    '''
    Convert region_ds9.region_pix into numpy masked array
    ------
    Parameters:
    region_pix: region_ds9.region_pix
        Can be converted from imported ds9 region using "region_sky.to_pixel(wcs)".
    data: np.2darray
    ------
    Returns: 
    data_region: numpy masked array, values in region are masked. 
    '''
    apmask = region_pix.to_mask()
    shape = data.shape
    mask=apmask.to_image(shape=((shape[0],shape[1])))
    ap_mask=mask==0
    data_region = np.ma.masked_where(ap_mask,data)

    return data_region

def casaMask_convert(data,region_masked):
    '''
    Convert the CASA .mask into numpy masked array
    ------
    Parameters:
    data: np.2darray
        2D image data
    region_masked: np.2darray
        2D array of 0 and 1 of the mask imported from .mask file using "fits_import()".  
    ------
    Returns: 
    data_region: numpy masked array, values in region are masked. 
    '''
#     data_mask=data_masked.mask
    region_mask=np.ma.make_mask(region_masked==0)
#     region_mask=np.ma.mask_or(data_mask,region_mask)
    data_region=np.ma.masked_where(region_mask,data_masked)
    return data_region

def Annulus_reg2phot(annulus_list):
    '''
    Convert the sky annlus in Regions class to a list of 
    sky apertures in photutils.aperture class
    '''
    annulus_list_out = [\
                        SkyEllipticalAnnulus(positions=annulus.center,
                                             a_in = annulus.inner_width/2,
                                             a_out = annulus.outer_width/2,
                                             b_in = annulus.inner_width/2,
                                             b_out = annulus.outer_width/2,
                                             theta = annulus.angle+math.pi/2*u.radian)\
                                             for annulus in annulus_list]
    return annulus_list_out

def Ellipse_reg2phot(aperture_list):
    '''
    Convert the sky annlus in Regions class to a list of 
    sky apertures in photutils.aperture class
    '''
    aperture_list_out = [\
                        SkyEllipticalAperture(positions=aperture.center,
                                             a = aperture.width/2,
                                             b = aperture.height/2, 
                                             theta = aperture.angle+math.pi/2*u.radian)\
                                             for aperture in aperture_list]
    return aperture_list_out


def flux_mask_get(data_region,rms,chans,chan_width):
    '''
    Measure a flux in a region defined by numpy mask.
    ------
    Parameters:
    data_region: np.masked.data
        numpy masked 2d array with values in regioon masked. 
    '''
    flux=np.ma.sum(data_region)/beam_area_pix
    chans_tmp=chans+np.zeros((np.shape(data_region)[0],np.shape(data_region)[1]))
    error=np.sqrt(chans_tmp)*rms*chan_width/sqrt(beam_area_pix)
    error_masked=np.ma.masked_where(data_region.mask,error)
    uncertainty=math.sqrt(np.ma.sum(np.power(error_masked,2)))
    return flux, uncertainty

def flux_aperture_get(data_masked,aperture,rms,chans,chan_width, beamarea_pix):
    '''
    Measure the flux in a region defined by photutils.aperture.
    '''
    data_cut=data_masked.data
    mask=data_masked.mask
    if np.shape(chans) == ():
        chans = np.full(data_cut.shape, chans) 
    flux=aperture_photometry(data_cut,apertures=aperture,mask=mask)['aperture_sum'][0]/beamarea_pix
    error=np.sqrt(chans)*rms*chan_width/np.sqrt(beamarea_pix)
    uncertainty=aperture_photometry(data_cut,apertures=aperture,mask=mask,error=error)['aperture_sum_err'][0]

    return flux, uncertainty

def group_pix_to_pix(data_in, wcs_in, data_tmpl, wcs_tmpl):
    '''
    Group the pixels of data_in by the pixels of data_tmpl 
    ------
    Parameters:
    data_in: np.2darray
        High-resolution image data that needs to be grouped into low 
        resolution pixels. 
    wcs_in: 
        Wcs for the input data. 
    data_tmpl: np.2darray
        Low-resolution template data to be matched to the input data.
    wcs_tmpl
        Wcs for the template data. 
    ------
    Return:
    matched_idx: np.mask
        Numpy mask array to determine if the smaller pixel in data_in belong
        to the larger pixel in data_tmpl.
    coords_label: np.1darray
        Numpy array of labels coorresponding to the pixel number in data_tmpl
        for each pixel in data_in. The pixel number is a 1d flattened array.
        The returned coords_label can then be used to bin the input data. 
    '''
    map_in_shape=np.shape(data_in)
    ny_in, nx_in=map_in_shape
    xs_in, ys_in = np.meshgrid(np.arange(nx_in), np.arange(ny_in))
    coords_in = pixel_to_skycoord(xs_in, ys_in, wcs_in)
    
    map_tmpl_shape = np.shape(data_tmpl)    
    ny_tmpl, nx_tmpl = map_tmpl_shape
    xs, ys = np.meshgrid(np.arange(nx_tmpl), np.arange(ny_tmpl))
    coords_tmpl=pixel_to_skycoord(xs, ys, wcs_tmpl)
    pixel_labels_tmpl = np.arange(xs.size)
    deltax, deltay = np.abs(wcs_tmpl.wcs.cdelt) * 3600

    pixel_map_arr = np.full((nx_in, ny_in), np.nan).flatten()

    i_in=0
    npix_in = coords_in.flatten().size
    dra, ddec = np.zeros(npix_in), np.zeros(npix_in)
    i_tmpl, d2d, d3d = match_coordinates_sky(coords_in.flatten(), coords_tmpl.flatten())
    dra, ddec = (coords_in.flatten()).spherical_offsets_to(
        coords_tmpl.flatten()[i_tmpl])
    dra = dra.arcsec
    ddec = ddec.arcsec

    good = (-deltax/2-0.001 <= dra) & (dra < deltax/2+0.001) & (-deltay/2-0.001 <= ddec) & (ddec < deltay/2+0.001)
    coords_labels = np.full(np.shape(coords_in),np.nan).flatten()

    coords_labels[good] = pixel_labels_tmpl[i_tmpl[good]]
    matched_idx = good

    return matched_idx, coords_labels


def bin_data(data_in, coords_labels, data_tmpl, weights=None):
    '''
    Bin the input data to have the same shape as the template data
    with larger pixel size. 
    ------
    Parameters:
    data_in: np.2darray
        High-resolution image data that needs to be binned. 
    coords_labels: np.1darray
        Results from function 'group_pix_to_pix()'. 
    data_tmpl: np.2darray
        Template 2d data to be matched with the same shape. 
    weights: np.2darray
        Weight for each pixel in the input data when doing 
        binning average. 
    ------
    Return:
    data_binned: np.2darray
        Binned data
    '''
    data_binned = np.full(np.shape(data_tmpl.flatten()), np.nan)

    for i in np.unique(coords_labels):
        if np.isnan(i):
            continue
        idx = int(i)
        condition = (coords_labels==i) &  (~np.isnan(data_in.flatten()))
        if len(np.where(condition)[0])>0:
            data_binned[idx] = np.average(data_in.flatten()[condition], weights=weights.flatten()[condition])

    data_binned = data_binned.reshape(np.shape(data_tmpl))

    return data_binned

###########################################################
# change the data wcs information

def reproj_binning(data, wcs, bin_num):
    '''
    bin the image in certain wcs frame.
    '''
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

def reproject_north(data,wcs):
    '''
    Parameters
    ----------
    data : np.2darray
        Data of the image
    wcs : wcs
        WCS information of the image

    Returns
    -------
    wcs_north : wcs
        Output wcs point to the north
    data_north : np.2darray
        Reprojected data

    '''
    wcs_north = WCS(naxis=2)
    wcs_north.wcs.crval = wcs.wcs.crval
    wcs_north.wcs.crpix = wcs.wcs.crpix
    cd = wcs.wcs.cd
    wcs_north.wcs.cd = np.sqrt(cd[0,0]**2+cd[1,0]**2)*np.array([[-1,0],[0,1]])
    wcs_north.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    data_north, footprint = reproject_exact((data, wcs), wcs_north, 
                                            shape_out=np.shape(data))
    
    return wcs_north, data_north

###########################################################
# Processing 3d data

def cut_3d(data,position,size,wcs):
    '''
    Cut the data cubes with 2D rectangle
    '''
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

# mask 3d cube with a region file in specified channels (June 28th)
def Regmask3d(data,region_pix,lowchan,highchan):
    '''

    '''
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

def make_mom0(dataCubes, pixsize=100, value=1e10):
    '''
    ------
    Parameters
    dataCubes: 3d numpy array
        3D numpy array to make moment maps
    pixsize: float
        Size of each pixel in pc
    value: float
        pixel unit in solar mass
    '''
    mom0 = np.nansum(dataCubes, axis=2) * value / pixsize**2
    
    return mom0
    
def make_mom1(dataCubes, vel_1d):
    '''
    ------
    Parameters 
    dataCubes: 3d numpy array
        3D numpy array to make moment maps
    vel_1d: 1d numpy array
        1D array of velocity values corresponding to the 3rd
        axis of the dataCubes. 
    '''          
    vel_3d = np.full(np.shape(dataCubes),fill_value=np.nan)
    vel_3d[:] = vel_1d
    mom1 = np.nansum(vel_3d*dataCubes,axis=2) / np.nansum(dataCubes,axis=2)
              
    return mom1

def make_mom2(dataCubes, vel_1d):
    '''
    ------
    Parameters
    dataCubes: 3d numpy array
        3D numpy array to make moment maps
    vel_1d: 1d numpy array
        1D array of velocity values corresponding to the 3rd
        axis of the dataCubes.
    '''
    vel_3d = np.full(np.shape(dataCubes),fill_value=np.nan)
    vel_3d[:] = vel_1d
    mom1 = np.nansum(vel_3d*dataCubes,axis=2) / np.nansum(dataCubes,axis=2)
    
    mom1_3d = np.repeat(mom1[:,:,np.newaxis], len(vel_1d), axis=2)
    vel_diff = vel_3d - mom1_3d
    mom2 = np.sqrt(np.nansum(vel_diff**2*dataCubes,axis=2) / np.nansum(dataCubes,axis=2))
    
    return mom2

def make_Tpeak(dataCubes, pixsize=100, value=1e10, alphaCO=4.3, ratio=0.7, deltaV=2):
    '''
    ------
    Parameters
    dataCubes: 3d numpy array
        3D numpy array to make moment maps
    pixsize: float
        Size of each pixel in pc
    value: float
        pixel unit in solarmass
    alphaCO: float
        CO-H2 conversion factor
    ratio: float
        CO 2-1/1-0 ratio
    deltaV : float
        Velocity resolution per channel
    '''
    mom8 = np.nanmax(dataCubes, axis=2)*value/pixsize**2
    Tpeak = mom8 / alphaCO / deltaV * ratio
    Tpeak[np.where(np.isnan(Tpeak))] = 0
    
    return Tpeak


    

