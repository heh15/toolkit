# Run this locally
# from ~/Dropbox/mac/wise_w3_vs_co
import glob
import astropy
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
from astropy import wcs
from astropy.wcs import WCS
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from astropy.table import Column

from astropy.cosmology import FlatLambdaCDM
from astropy import coordinates
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS
from ltsfit.lts_linefit import lts_linefit
import scipy
from scipy import stats
import itertools

from PyPDF2 import PdfFileMerger

from reproject import reproject_interp
from reproject import reproject_exact

import sklearn
from sklearn.decomposition import PCA

from skimage.transform import rescale
from skimage.transform import resize
from skimage.measure import block_reduce

import pandas as pd
import pickle
import scipy
import scipy.stats
from scipy.optimize import curve_fit

import linmix
import reproject_califa as reproj_c

fname_surf_dens_all_dict = '/Users/ryan/Dropbox/mac/wise_w3_vs_co/surf_dens_all_v2.pk'  # Without Aniano kernels, until that is fixed, and with more conservative s/n cuts in mom0 map making
surf_dens_all = pickle.load(open(fname_surf_dens_all_dict, 'rb'))

zomgs = ascii.read(
    '/Users/ryan/Dropbox/mac/wise_w3_vs_co/z0mgs_edge_match.txt')

edge_detable = pd.read_csv(
    '/usr/local/lib/python3.7/site-packages/edge_pydb/dat_glob/external/build/DETableFinal.csv'
)

edge_detable_names = np.array(edge_detable['Name'])
morphs = np.array(edge_detable[' ledaMorph'])
# etypes = (morphs == ' E') | (morphs == ' E-S0') |  (morphs == ' S0') |  (morphs == ' S0-a')
# det = np.array(edge_detable[' coSmooth']).astype(float)/np.array(edge_detable[' coeSmooth']).astype(float) >3
# pl.scatter(np.array(edge_detable[' caSFR']).astype(float)[~etypes], (np.log10(np.array(edge_detable[' coMmol']).astype(float))-np.array(edge_detable[' caMass']).astype(float))[~etypes])
# pl.scatter(np.array(edge_detable[' caSFR']).astype(float)[etypes], (np.log10(np.array(edge_detable[' coMmol']).astype(float))-np.array(edge_detable[' caMass']).astype(float))[etypes],  c='r')
# pl.scatter( np.array(edge_detable[' caSFR']).astype(float)[~det],(np.log10(np.array(edge_detable[' coMmol']).astype(float))-np.array(edge_detable[' caMass']).astype(float))[~det], c='k', facecolors='none', edgecolors='k', marker='s', s=3)
# loglco_edge = np.log10(np.array(edge_detable[' coMmol']).astype(float) / 4.36)
edge_detable_dist = np.array(edge_detable[' caDistMpc'])
edge_detable_z = np.array(edge_detable[' caZgas'])

logmmol_edge = np.log10(
    np.array(edge_detable[' coMmol']).astype(float) / 4.36 * 4.35)
logmmol_err_edge = 0.434 * np.array(
    edge_detable[' coeMmol']).astype(float) / np.array(
        edge_detable[' coMmol']).astype(float)

log_sfr_edge = np.array(
    edge_detable[' caLHacorr']).astype(float) + np.log10(7.9e-42)
log_mstar_edge = np.array(edge_detable[' caMstars']).astype(float)
ur_color_edge = np.array(edge_detable[' Su']) - np.array(edge_detable[' Sr'])
metallicity_edge = np.array(edge_detable[' caOH_O3N2'])

leda = Table.read(
    '/usr/local/lib/python3.7/site-packages/edge_pydb/dat_glob/external/edge_leda.csv',
    format='ascii.ecsv')
edge_califa = Table.read(
    '/usr/local/lib/python3.7/site-packages/edge_pydb/dat_glob/external/edge_califa.csv',
    format='ascii.ecsv')
edge_califa_names = np.array(edge_califa['Name'])


def get_n_odd(n):
    n_odd = [1]
    dn = 2
    i = 1
    while len(n_odd) < n:
        n_odd.append(n_odd[i - 1] + dn)
        i += 1
    return n_odd


def get_3color(galname, ra, dec):
    w = wcs.WCS(naxis=2)
    gal = galname
    if galname == 'NGC4211NED02':
        gal = 'NGC4211B'
    # c = coordinates.get_icrs_coordinates(gal) # center of galaxy coordinate
    linwcs = lambda x, y, n: ((x - y) / n, (x + y) / 2)
    cdeltaX, crvalX = linwcs(ra - 1.2 / 60. / 2., ra + 1.2 / 60. / 2., 512)
    cdeltaY, crvalY = linwcs(dec - 1.2 / 60. / 2., dec + 1.2 / 60. / 2., 512)
    # what is the center pixel of the XY grid.
    w.wcs.crpix = [512. / 2, 512. / 2]
    # what is the galactic coordinate of that pixel.
    w.wcs.crval = [crvalX, crvalY]
    # what is the pixel scale in lon, lat.
    w.wcs.cdelt = np.array([cdeltaX, cdeltaY])
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    # ax = pl.subplot(2,5,1,projection=w)
    image = mpimg.imread(
        glob.glob('/Users/ryan/venus/home/edge_sdss_thumbnails/%s*.jpg' %
                  (galname, ))[0])
    return image, w


# edge_califa = Table.read(
#     '/usr/local/lib/python3.7/site-packages/edge_pydb/dat_glob/external/edge_califa.csv',
#     format='ascii.ecsv')
# edge_names = np.array(list(edge_califa['Name']))
# # mstar_edge = np.array(edge_califa['caMass'])
# mstar_edge = np.array(edge_califa['caMstars'])
# metallicity_edge = np.array(edge_califa['caOH_O3N2'])
# ur_color_edge = np.array(edge_califa['Su']) - np.array(edge_califa['Sr'])

mstar_bins_default = [[9.5, 10], [10, 10.5], [10.5, 11], [11, 11.6]]
sfr_bins_default = [[-1.5, -0.5], [-.5, 0], [0, .5], [.5, 2]]
ur_bins_default = [[.8, 1.5], [1.5, 1.75], [1.75, 2], [2, 2.25], [2.25, 2.7]]
met_bins_default = [[8.4, 8.45], [8.45, 8.5], [8.5, 8.55], [8.55, 8.6]]


class FitAndGlobalPropertiesFullSample(object):
    # TODO (Fri Nov 1):
    # - Make a way to easily add global properties to the "data" dict in the
    # function above (plot_sigma_fit_params_vs_global_logmstar). Right now it just has
    # logm, sfr, metallicity, u-r and interacting/isolated status, but if I want to add
    # more, I have to hard-code it into the loop in that function. This will help
    # relieve pain when I haven't looked at the code for a while and forget how
    # to add new properties.
    #
    # How I imagine this will be used:
    #   Start with an already-made "data" dictionary
    #   Then use the "add_property" function to add new global properties.
    #
    # - Also, will need to change the above function to return a FitAndGlobalPropertiesFullSample
    # object, not just a dictionary.
    #
    # - Will also need to make sure the rest of the functions which use the "data"
    # dictionary still work with the FitAndGlobalPropertiesFullSample object.
    #
    # - Finally, this class definition should go closer to the top of this script.
    #
    # - Consider writing classes for any other types of dictionaries. It is a
    # bit confusing having multiple dictionary types.
    def __init__(self, data):
        self.data = data
        self.data_fits = dict()
        # Defualt properties
        self.global_properties = [
            'logm_global', 'sfr_global', 'met_global', 'ur_global'
        ]

        self.global_property_bins = dict()
        self.global_property_bins['logm_global'] = mstar_bins_default
        self.global_property_bins['sfr_global'] = sfr_bins_default
        self.global_property_bins['met_global'] = met_bins_default
        self.global_property_bins['ur_global'] = ur_bins_default

        self.global_property_labels = dict()
        self.global_property_labels['logm_global'] = r"$\log \> M_*/M_\odot$"
        self.global_property_labels[
            'sfr_global'] = r"$\log \> \mathrm{SFR}/M_\odot\>\mathrm{yr^{-1}}$"
        self.global_property_labels['met_global'] = r"$12+\log(\mathrm{O/H})$"
        self.global_property_labels['ur_global'] = r"$u-r$"
        self.pixel_catalog = None

    def get_pixel_catalog(self, return_catalog=True):
        '''
        Create a catalog of pixel properties for the whole sample.

        Property descriptions:
            id (int) : Just an integer label (starting from 0)
            loglco (float) : log10(CO lum. in K km/s pc^2)
            err_loglco (float) : Uncertainty in the above
            logl12 (float) : log10(12um lum. in Lsun)
            err_logl12 (float) : Uncertainty in the above
            galname (str) : Name of the galaxy this pixel belongs to
            pix_area (float) : Area of the pixel in pc^2
            met (float) : Gas-phase metallicity (12+log(O/H))
            alpha_co (float) : CO-to-H2 conversion factor from this metallicity
            log_sigma_mstar (float) : log10(stellar mass surface density in Msun/pc^2)
            log_sigma_sfr (float) : log10(SFR surface density in Msun/yr/pc^2)
            bpt (int) : BPT classification. (-1=SF, 0=comp., 1=LIER, 2=Seyfert)

        '''
        pix_cat = {
            'log_lco_aco3p2_all': [],
            'err_log_lco_aco3p2_all': [],
            'log_lco_aco3p2_sf': [],
            'err_log_lco_aco3p2_sf': [],
            'log_lco_aco_met': [],
            'err_log_lco_aco_met': [],
            'log_sigh2_aco3p2_all': [],
            'err_log_sigh2_aco3p2_all': [],
            'log_sigh2_aco3p2_sf': [],
            'err_log_sigh2_aco3p2_sf': [],
            'log_sigh2_aco_met': [],
            'err_log_sigh2_aco_met': [],
            'log_l12': [],
            'err_log_l12': [],
            'log_sig12': [],
            'err_log_sig12': [],
            'met': [],
            'alpha_co': [],
            'log_sigmstar': [],
            'log_sigsfr': [],
            'err_log_sigsfr': [],
            'bpt': [],
            'id': [],
            'pix_area': [],
            'galname': []
        }

        bad_value = np.nan
        alpha_co = 3.2

        id = 0
        for galname in list(surf_dens_all.keys()):
            if len(
                    glob.glob(
                        '/Users/ryan/venus/shared_data/califa/DR3-stack/%s/%s_result.pk'
                        % (galname, galname))) == 0:
                print("Nothing exists for this galaxy")
                continue

            if galname not in self.data['name']:
                continue

            pixel_area_pc2 = surf_dens_all[galname]['pix_area_pc2']

            # Note: swapped x and y. "x" in gal_dict is H2, "y" is 12um.
            y_fit = gal_dict[galname][
                'x_fit']  # Sigma H2 (alpha_co=3.2 assumed)
            y_err_fit = gal_dict[galname]['x_err_fit']
            x_fit = gal_dict[galname]['y_fit']  # Sigma 12um
            x_err_fit = gal_dict[galname]['y_err_fit']

            bpt = self.map_bpt(galname, return_map=True, plot_map=False).flatten()
            # bpt = bpt.astype(int).flatten()  # Make sure this is integer type
            # Make sure no invalid values (inf, or integers outside acceptable range)

            alpha_co_map = self.map_alpha_co(galname,
                                             which_alpha='met_alpha_sf_only',
                                             return_map=True,
                                             plot_map=False).flatten()

            sigma_h2_map = self.map_sigma_h2(galname,
                                             which_alpha='fixed',
                                             return_map=True,
                                             plot_map=False).flatten()
            sigma_h2_map_sf = self.map_sigma_h2(galname,
                                                which_alpha='fixed_sf_only',
                                                return_map=True,
                                                plot_map=False).flatten()
            sigma_h2_map_metalpha = self.map_sigma_h2(
                galname,
                which_alpha='met_alpha_sf_only',
                return_map=True,
                plot_map=False).flatten()
            err_sigma_h2_map = self.map_sigma_h2_err(galname,
                                                     which_alpha='fixed',
                                                     return_map=True,
                                                     plot_map=False,
                                                     linear=False).flatten()
            err_sigma_h2_map_sf = self.map_sigma_h2_err(
                galname,
                which_alpha='fixed_sf_only',
                return_map=True,
                plot_map=False,
                linear=False).flatten()
            err_sigma_h2_map_metalpha = self.map_sigma_h2_err(
                galname,
                which_alpha='met_alpha_sf_only',
                return_map=True,
                plot_map=False,
                linear=False).flatten()

            sigma_12_map = self.map_sigma_12um(galname,
                                               return_map=True,
                                               plot_map=False).flatten()
            err_sigma_12_map = self.map_sigma_12um_err(galname,
                                                       return_map=True,
                                                       plot_map=False,
                                                       linear=False).flatten()

            sigma_mstar_map = pickle.load(
                open(reproj_c.fname_mstar_stacked(galname),
                     'rb'))['mstar'].flatten()
            sigma_mstar_map = np.log10(sigma_mstar_map)

            # Sigma SFR (Msun/yr/kpc^2) ** note not pc^2 **
            sigma_sfr_dict = pickle.load(
                open(reproj_c.fname_halpha_stacked(galname), 'rb'))
            sigma_sfr_map = sigma_sfr_dict['sigma_sfr'].flatten()
            err_sigma_sfr_map = sigma_sfr_dict['sigma_sfr_err'].flatten()
            err_sigma_sfr_map = 0.434 * err_sigma_sfr_map / sigma_sfr_map
            sigma_sfr_map = np.log10(sigma_sfr_map)

            metallicity = pickle.load(
                open(reproj_c.fname_metallicity_stacked(galname),
                     'rb'))['metallicity'].flatten()

            # Make sure co_noise_maps is imported.
            co_rms = cnoise.calculate_noise_map(galname) # RMS per pixel in Jy/beam
            # convert this to a 5 sigma upper limit (5 * sigma in linear units, not log)
            

            npix = sigma_12_map.size
            # Loop over all pixels for this galaxy
            for j in range(npix):
                # CO luminosities
                # alpha_co = 3.2, all pix.
                pix_cat['log_lco_aco3p2_all'].append(sigma_h2_map[j] +
                                                     np.log10(pixel_area_pc2 /
                                                              alpha_co))
                pix_cat['err_log_lco_aco3p2_all'].append(
                    err_sigma_h2_map[j])

                # alpha_co = 3.2, sf pix.
                pix_cat['log_lco_aco3p2_sf'].append(sigma_h2_map_sf[j] +
                                                    np.log10(pixel_area_pc2 /
                                                             alpha_co))
                pix_cat['err_log_lco_aco3p2_sf'].append(err_sigma_h2_map_sf[j])

                # alpha_co(metallicity)
                pix_cat['log_lco_aco_met'].append(sigma_h2_map_metalpha[j] +
                                                  np.log10(pixel_area_pc2 /
                                                           alpha_co))
                pix_cat['err_log_lco_aco_met'].append(
                    err_sigma_h2_map_metalpha[j])

                # H2 surface densities
                # alpha_co = 3.2, all pix.
                pix_cat['log_sigh2_aco3p2_all'].append(sigma_h2_map[j])
                pix_cat['err_log_sigh2_aco3p2_all'].append(
                    err_sigma_h2_map[j])

                # alpha_co = 3.2, sf pix.
                pix_cat['log_sigh2_aco3p2_sf'].append(sigma_h2_map_sf[j])
                pix_cat['err_log_sigh2_aco3p2_sf'].append(err_sigma_h2_map_sf[j])

                # alpha_co(metallicity)
                pix_cat['log_sigh2_aco_met'].append(sigma_h2_map_metalpha[j])
                pix_cat['err_log_sigh2_aco_met'].append(
                    err_sigma_h2_map_metalpha[j])

                # CO RMS


                # 12um luminosity
                pix_cat['log_l12'].append(sigma_12_map[j] +
                                          np.log10(pixel_area_pc2))
                pix_cat['err_log_l12'].append(err_sigma_12_map[j])
                # 12um surface density
                pix_cat['log_sig12'].append(sigma_12_map[j])
                pix_cat['err_log_sig12'].append(err_sigma_12_map[j])

                pix_cat['met'].append(metallicity[j])
                pix_cat['alpha_co'].append(alpha_co_map[j])
                pix_cat['log_sigmstar'].append(sigma_mstar_map[j])
                pix_cat['log_sigsfr'].append(sigma_sfr_map[j])
                pix_cat['err_log_sigsfr'].append(err_sigma_sfr_map[j])
                pix_cat['bpt'].append(bpt[j])

                pix_cat['id'].append(id)
                pix_cat['pix_area'].append(pixel_area_pc2)
                pix_cat['galname'].append(galname)
                id += 1

        # Convert lists to arrays
        for k in pix_cat.keys():
            pix_cat[k] = np.hstack(np.array(pix_cat[k])).flatten()

        self.pixel_catalog = pix_cat

        if return_catalog:
            return pix_cat

    def add_property(self,
                     property_name,
                     property_getter,
                     property_label='',
                     property_bins=None):
        '''
        Args:
            property_name (str)
            property_getter : function that takes as input a galaxy name
                and outputs the property (can be any type of property).
            property_label (str) : latex label (for axes)
        '''
        property_arr = []
        for galname in list(surf_dens_all.keys()):
            if len(
                    glob.glob(
                        '/Users/ryan/venus/shared_data/califa/DR3-stack/%s/%s_result.pk'
                        % (galname, galname))) == 0:
                print("Nothing exists for this galaxy")
                continue

            if galname not in self.data['name']:
                continue

            property_arr.append(property_getter(galname))

        property_arr = np.array(property_arr)

        # Update
        self.data[property_name] = property_arr
        self.global_properties.append(property_name)
        self.global_property_labels[property_name] = property_label
        if property_bins is not None:
            self.global_property_bins[property_name] = property_bins
        else:
            property_arr_temp = property_arr[(~np.isinf(property_arr))
                                             & (~np.isnan(property_arr))]
            self.global_property_bins[property_name] = [[
                property_arr_temp.min(),
                property_arr_temp.max()
            ]]

    def add_property_bins(self, property_name, property_bins):
        if property_name not in self.global_properties:
            print("Error: %s not in global properties" % (property_name, ))
        else:
            self.property_bins[property_name] = property_bins

    def add_property_label(self, property_name, property_label):
        if property_name not in self.global_properties:
            print("Error: %s not in global properties" % (property_name, ))
        else:
            self.global_property_labels[property_name] = property_label

    def plot_sigma_fit_params_vs_other_property(self,
                                                which_fit='linmix_fit_reverse',
                                                which_alpha='fixed',
                                                figsize=None,
                                                marker_hubble_types=None,
                                                calc_pearson=None,
                                                fontsize_axes=10):
        '''
        (optional) marker_hubble_types (dict): marker_hubble_types['S'] = '*',
            for example. Marker will be overridden if multiple_gal == 'M' or
            dont_fit == True.
        (optional) calc_pearson (dict): calc_pearson['S'] = fun(galname),
            for example. Will calculate Pearson-r for spiral galaxies,
            and plot in each panel. The function "fun" takes a galaxy
            name as input and returns True or False if it is 'S' or not.
        '''
        gals_to_label = []
        if figsize is None:
            pl.figure(figsize=(8, 12))
        else:
            pl.figure(figsize=figsize)
        labelled_mergers = False
        labelled_exclude = False
        labelled_gals = False
        n_properties = len(self.global_properties)
        n_odd = get_n_odd(n_properties)

        labels_row = [
            self.global_property_labels[prop]
            for prop in self.global_property_labels.keys()
        ]

        if calc_pearson is not None:
            pearson_slopes = dict()
            pearson_intercepts = dict()
            for pcalc in calc_pearson.keys():
                pearson_slopes[pcalc] = dict()
                pearson_intercepts[pcalc] = dict()
                for p in self.global_properties:
                    pearson_slopes[pcalc][p] = {'x': [], 'y': [], 'r': 0}
                    pearson_intercepts[pcalc][p] = {'x': [], 'y': [], 'r': 0}

        for galname in list(surf_dens_all.keys()):
            if len(
                    glob.glob(
                        '/Users/ryan/venus/shared_data/califa/DR3-stack/%s/%s_result.pk'
                        % (galname, galname))) == 0:
                print("Nothing exists for this galaxy")
                continue

            gal_dict_i = gal_dict[galname]
            pixel_area_pc2 = surf_dens_all[galname]['pix_area_pc2']

            dont_fit = False
            if galname in ['NGC5406', 'NGC2916', 'UGC09476']:
                dont_fit = True

            if which_fit.split('_')[0] == 'linmix':
                check_type = dict
            if which_fit.split('_')[0] == 'lts':
                check_type = lts_linefit
            if which_fit.split('_')[0] == 'fit':
                check_type = dict

            # Which alpha_co do you want?
            if which_alpha == 'fixed':
                fit_i = gal_dict_i
                x_fit = gal_dict[galname]['x_fit']
                x_err_fit = gal_dict[galname]['x_err_fit']
                y_fit = gal_dict[galname]['y_fit']
                y_err_fit = gal_dict[galname]['y_err_fit']
            if which_alpha == 'fixed_sf_only':
                fit_i = gal_dict_i['fits_fixed_alpha_sf_only']
                x_fit = gal_dict[galname]['x_fit_sf']
                x_err_fit = gal_dict[galname]['x_fit_sf_err']
                y_fit = gal_dict[galname]['y_fit_sf']
                y_err_fit = gal_dict[galname]['y_fit_sf_err']
            if which_alpha == 'met_alpha_sf_only':
                fit_i = gal_dict_i['result_met_alpha_sf_only']
                x_fit = gal_dict[galname]['x_fit_met']
                x_err_fit = gal_dict[galname]['x_fit_met_err']
                y_fit = gal_dict[galname]['y_fit_met']
                y_err_fit = gal_dict[galname]['y_fit_met_err']

            if (type(fit_i[which_fit]) == check_type):
                if check_type == dict:
                    slope_best = fit_i[which_fit]['slope']
                    intercept_best = fit_i[which_fit]['intercept']
                    slope_err = fit_i[which_fit]['slope_err']
                    intercept_err = fit_i[which_fit]['intercept_err']
                    if which_fit.split('_')[-1] == 'masked':
                        if np.where(fit_i['lts_fit_mask'] == True)[0].size < 6:
                            continue
                else:
                    intercept_best, slope_best = fit_i[which_fit].ab
                    intercept_err, slope_err = fit_i[which_fit].ab_err

                multiple_gal = get_inter_califa(galname)
                marker = 'o'
                ecolor = 'b'
                if marker_hubble_types is not None:
                    # Marker will be overridden if multiple_gal == 'M' or dont_fit == True
                    hubble, hubble_min, hubble_max = get_hubble_type_califa(
                        galname)
                    if (hubble == hubble_min) and (hubble == hubble_max):
                        # Must be absolutely certain about hubble type (S, I or E)
                        marker = marker_hubble_types[hubble]

                label = None
                if (dont_fit == False) and (labelled_gals == False) and (
                        multiple_gal != 'M'):
                    label = 'Indiv. galaxies'
                    labelled_gals = True
                if (dont_fit == True) and (labelled_exclude == False) and (
                        multiple_gal != 'M'):
                    label = 'Excluded'
                    labelled_exclude = True
                if (dont_fit == False) and (multiple_gal == 'M') and (
                        labelled_mergers == False):
                    label = 'Pair/merger'
                    labelled_mergers = True

                props_row = [(self.data[prop])[self.data['name'] == galname][0]
                             for prop in self.global_properties]

                if calc_pearson is not None:
                    # Decide whether or not to include these points
                    # in Pearson-r calculation
                    add_to_pearson = dict()
                    for pcalc in calc_pearson.keys():
                        classify_for_pearson = calc_pearson[pcalc]
                        add_to_pearson[pcalc] = classify_for_pearson(galname)
                        if add_to_pearson[pcalc] == False:
                            marker = 'o'

                if multiple_gal == 'M':
                    marker = 's'
                    ecolor = 'r'
                if dont_fit == True:
                    ecolor = 'g'
                    marker = 'x'

                k = 0
                for j in n_odd:
                    global_property_name_temp = self.global_properties[k]
                    # First column: slope
                    pl.subplot(n_properties, 2, j)
                    scatterplot(props_row[k],
                                slope_best,
                                labels_row[k],
                                r"Best-fit $N$",
                                yerr=slope_err,
                                marker=marker,
                                ecolor=ecolor,
                                label=label,
                                alpha=0.7,
                                fontsize_axes=fontsize_axes)

                    if calc_pearson is not None:
                        for pcalc in calc_pearson.keys():
                            if add_to_pearson[pcalc] == True:
                                pearson_slopes[pcalc][
                                    global_property_name_temp]['x'].append(
                                        props_row[k])
                                pearson_slopes[pcalc][
                                    global_property_name_temp]['y'].append(
                                        slope_best)

                    if galname in gals_to_label:
                        if galname != 'ARP220':
                            pl.text(props_row[k],
                                    slope_best,
                                    galname,
                                    fontsize=8,
                                    weight='bold')
                        else:
                            nn = gals_to_label.index(galname)
                            pl.annotate(galname,
                                        xy=(props_row[k], slope_best),
                                        weight='bold',
                                        xytext=(props_row[k] - .5,
                                                -0.5 + 0.1 * nn),
                                        fontsize=8,
                                        arrowprops=dict(arrowstyle="->",
                                                        color='k'))
                    # Second column: intercept
                    pl.subplot(n_properties, 2, j + 1)
                    scatterplot(props_row[k],
                                intercept_best,
                                labels_row[k],
                                r"Best-fit $\log \> C$",
                                yerr=intercept_err,
                                marker=marker,
                                ecolor=ecolor,
                                label=label,
                                alpha=0.7,
                                fontsize_axes=fontsize_axes)

                    if calc_pearson is not None:
                        for pcalc in calc_pearson.keys():
                            if add_to_pearson[pcalc] == True:
                                pearson_intercepts[pcalc][
                                    global_property_name_temp]['x'].append(
                                        props_row[k])
                                pearson_intercepts[pcalc][
                                    global_property_name_temp]['y'].append(
                                        intercept_best)

                    pl.ylim(-2, 2)
                    if galname in gals_to_label:
                        if galname != 'ARP220':
                            pl.text(props_row[k],
                                    intercept_best,
                                    galname,
                                    fontsize=8,
                                    weight='bold')
                        else:
                            nn = gals_to_label.index(galname)
                            pl.annotate(galname,
                                        xy=(props_row[k] - .5, intercept_best),
                                        weight='bold',
                                        xytext=(props_row[k], -1 + 0.1 * nn),
                                        fontsize=8,
                                        arrowprops=dict(arrowstyle="->",
                                                        color='k'))
                    k += 1

        # Now calculate Pearson-r for each row (galaxy property)
        # and column (slope and intercept)
        pearson_result = dict()
        pearson_result['slope'] = dict()
        pearson_result['intercept'] = dict()

        i = 1
        if calc_pearson is not None:
            for pcalc in calc_pearson.keys():
                for global_property_name_temp in self.global_properties:
                    x = np.array(
                        pearson_slopes[pcalc][global_property_name_temp]['x'])
                    y = np.array(
                        pearson_slopes[pcalc][global_property_name_temp]['y'])
                    good = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isinf(x)) & (
                        ~np.isinf(y))
                    r = scipy.stats.pearsonr(x[good], y[good])
                    r_spearman = scipy.stats.spearmanr(x[good], y[good])
                    kendall_tau = scipy.stats.kendalltau(x[good], y[good])
                    pearson_slopes[pcalc][global_property_name_temp]['r'] = r
                    pl.subplot(n_properties, 2, i)
                    pl.title(
                        r"Pears./Spear. $r=%2.2f/%2.2f$, Kendall $\tau=%2.2f$"
                        % (r[0], r_spearman[0], kendall_tau[0]),
                        fontsize=7)

                    pearson_result['slope'][global_property_name_temp] = {
                        'label':
                        self.global_property_labels[global_property_name_temp],
                        'pearson':
                        r,
                        'spearman':
                        r_spearman,
                        'kendall':
                        kendall_tau
                    }

                    x = np.array(pearson_intercepts[pcalc]
                                 [global_property_name_temp]['x'])
                    y = np.array(pearson_intercepts[pcalc]
                                 [global_property_name_temp]['y'])
                    good = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isinf(x)) & (
                        ~np.isinf(y))
                    r = scipy.stats.pearsonr(x[good], y[good])
                    r_spearman = scipy.stats.spearmanr(x[good], y[good])
                    kendall_tau = scipy.stats.kendalltau(x[good], y[good])
                    pearson_intercepts[pcalc][global_property_name_temp][
                        'r'] = r
                    i += 1
                    pl.subplot(n_properties, 2, i)
                    pl.title(
                        r"Pears./Spear. $r=%2.2f/%2.2f$, Kendall $\tau=%2.2f$"
                        % (r[0], r_spearman[0], kendall_tau[0]),
                        fontsize=7)

                    pearson_result['intercept'][global_property_name_temp] = {
                        'label':
                        self.global_property_labels[global_property_name_temp],
                        'pearson':
                        r,
                        'spearman':
                        r_spearman,
                        'kendall':
                        kendall_tau
                    }
                    i += 1
            return pearson_result

    def get_mean_params_without_gal(self, galname, property_name):
        res = dict()
        nbins = len(self.global_property_bins[property_name])
        data = self.data

        param_key = 'mean_' + property_name + '_bin'

        for i in range(0, nbins):
            print(i)
            param_lo = param_bins[i][0]
            param_hi = param_bins[i][1]

            i_param_bin = (data['dont_fit'] == False) & (
                data[property_name] >= param_lo
            ) & (data[property_name] < param_hi) & (data['interacting'] != 'M')

            names_bin = data['name'][i_param_bin]
            if galname in names_bin:
                i_param_bin = (data['name'] !=
                               galname) & (data['dont_fit'] == False) & (
                                   data[property_name] >= param_lo) & (
                                       data[property_name] <
                                       param_hi) & (data['interacting'] != 'M')

                param_bin = data[property_name][i_param_bin]
                err_param_bin = np.ones(
                    param_bin.size) * 0.15  # assume 0.15 dex mstar uncertainty

                slope_bin = data['slope'][i_param_bin]
                slope_err_bin = data['slope_err'][i_param_bin]
                intercept_bin = data['intercept'][i_param_bin]
                intercept_err_bin = data['intercept_err'][i_param_bin]

                res['mean_slope'] = np.average(slope_bin)
                res['median_slope'] = np.median(slope_bin)
                res['std_slope'] = np.std(slope_bin) / np.sqrt(slope_bin.size)

                res['mean_intercept'] = np.average(intercept_bin)
                res['median_intercept'] = np.median(intercept_bin)
                res['std_intercept'] = np.std(intercept_bin) / np.sqrt(
                    intercept_bin.size)
                # res[param_key][i] = np.average(param_bin)
                return res

    def get_fit_params_binned_by_property(self, property_name='logm_global'):
        data = self.data
        param_bins = self.global_property_bins[property_name]
        res = dict()
        nbins = len(param_bins)
        res['mean_slope'] = np.zeros(nbins)
        res['median_slope'] = np.zeros(nbins)
        res['std_slope'] = np.zeros(nbins)
        res['mean_intercept'] = np.zeros(nbins)
        res['median_intercept'] = np.zeros(nbins)
        res['std_intercept'] = np.zeros(nbins)

        # res['n_logmstar_fit'] = {'low_mstar': dict(), 'high_mstar': dict()}
        # res['logc_logmstar_fit'] = {'low_mstar': dict(), 'high_mstar': dict()}

        res['fit_pixels_in_bin'] = [dict()] * nbins
        res['fit_xy'] = [[]] * nbins
        param_key = 'mean_' + property_name + '_bin'
        res[param_key] = np.zeros(nbins)

        for i in range(0, nbins):
            print(i)
            param_lo = param_bins[i][0]
            param_hi = param_bins[i][1]
            i_param_bin = (data['dont_fit'] == False) & (
                data[property_name] >= param_lo
            ) & (data[property_name] < param_hi) & (data['interacting'] != 'M')

            param_bin = data[property_name][i_param_bin]
            err_param_bin = np.ones(
                param_bin.size) * 0.15  # assume 0.15 dex mstar uncertainty

            slope_bin = data['slope'][i_param_bin]
            slope_err_bin = data['slope_err'][i_param_bin]
            intercept_bin = data['intercept'][i_param_bin]
            intercept_err_bin = data['intercept_err'][i_param_bin]

            res['mean_slope'][i] = np.average(slope_bin)
            res['median_slope'][i] = np.median(slope_bin)
            res['std_slope'][i] = np.std(slope_bin) / np.sqrt(slope_bin.size)

            res['mean_intercept'][i] = np.average(intercept_bin)
            res['median_intercept'][i] = np.median(intercept_bin)
            res['std_intercept'][i] = np.std(intercept_bin) / np.sqrt(
                intercept_bin.size)

            res[param_key][i] = np.average(param_bin)
        self.data_fits[property_name] = res
        return res

    def plot_data_fits(self,
                       property_name='logm_global',
                       i=1,
                       j=2,
                       k=1,
                       legend=True):

        param_key = 'mean_' + property_name + '_bin'
        data_fits = self.data_fits[property_name]
        param_bin_centers = data_fits[param_key]

        pl.subplot(i, j, k)
        pl.errorbar(param_bin_centers,
                    data_fits['mean_slope'],
                    yerr=data_fits['std_slope'],
                    marker='s',
                    color='k',
                    markeredgecolor='k',
                    markerfacecolor='xkcd:goldenrod',
                    linewidth=1.5,
                    alpha=1,
                    label='Mean',
                    capsize=2,
                    markersize=5,
                    ecolor='k',
                    elinewidth=1,
                    markeredgewidth=1)
        pl.scatter(param_bin_centers,
                   data_fits['median_slope'],
                   marker='^',
                   edgecolor='k',
                   facecolor='xkcd:lime',
                   linewidth=0.5,
                   label='Median')

        pl.subplot(i, j, k + 1)
        pl.errorbar(param_bin_centers,
                    data_fits['mean_intercept'],
                    yerr=data_fits['std_intercept'],
                    marker='s',
                    color='k',
                    markeredgecolor='k',
                    markerfacecolor='xkcd:goldenrod',
                    linewidth=1.5,
                    alpha=1,
                    label='Mean',
                    capsize=2,
                    markersize=5,
                    ecolor='k',
                    elinewidth=1,
                    markeredgewidth=1)
        pl.scatter(param_bin_centers,
                   data_fits['median_intercept'],
                   marker='^',
                   edgecolor='k',
                   facecolor='xkcd:lime',
                   linewidth=0.5,
                   label='Median',
                   alpha=1.)
        if legend:
            pl.legend(loc='best')

    def plot_rgb(self, galname, ax=None, subplot=None):
        '''
        '''
        # Plot the map
        mdat, mhdr = fits.getdata(
            '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
            % (galname, ),
            header=True)
        w = wcs.WCS(mhdr)
        idx = np.where(leda['Name'] == galname)[0][0]
        ractr = leda['ledaRA'].quantity[idx].to(u.deg).value
        dcctr = leda['ledaDE'].quantity[idx].to(u.deg).value

        sdss_3color = get_3color(galname, ractr, dcctr)
        if ax is None:
            ax = pl

        if subplot is None:
            ax = pl.subplot(111, projection=sdss_3color[1])
        else:
            ax = pl.subplot(subplot[0],
                            subplot[1],
                            subplot[2],
                            projection=sdss_3color[1])
        ax = pl.gca()
        ra, dec = ax.coords[0], ax.coords[1]
        ra.set_ticklabel_visible(False)
        dec.set_ticklabel_visible(False)
        pl.imshow(sdss_3color[0][::-1, ::-1])

        # ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_autoscale_on(False)

        return

    def plot_sf_mask(self, galname, ax=None, subplot=None):
        '''
        '''
        return

    def map_bpt(self,
                galname,
                ax=None,
                subplot=None,
                return_map=False,
                plot_map=False,
                colorbar=True):
        '''
        Return and/or plot BPT
        '''

        if ax is None:
            ax = pl

        bpt_map = reproj_c.bpt_stacked(galname)

        if plot_map:
            # Plot the map
            mdat, mhdr = fits.getdata(
                '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
                % (galname, ),
                header=True)
            w = wcs.WCS(mhdr)
            idx = np.where(leda['Name'] == galname)[0][0]
            ractr = leda['ledaRA'].quantity[idx].to(u.deg).value
            dcctr = leda['ledaDE'].quantity[idx].to(u.deg).value

            x, y = w.wcs_world2pix(ractr, dcctr, 0)
            wcs_out = wcs.WCS(naxis=2)
            wcs_out.wcs.crpix = [x, y]
            # what is the galactic coordinate of that pixel.
            wcs_out.wcs.crval = [ractr, dcctr]
            # what is the pixel scale in lon, lat.
            wcs_out.wcs.cdelt = np.array([mhdr['CDELT1'], mhdr['CDELT2']])
            wcs_out.wcs.ctype = ['RA---SIN', 'DEC--SIN']
            if subplot is None:
                ax = pl.subplot(111, projection=wcs_out)
            else:
                ax = pl.subplot(subplot[0],
                                subplot[1],
                                subplot[2],
                                projection=wcs_out)
            ax = pl.gca()
            ra, dec = ax.coords[0], ax.coords[1]
            ra.set_ticklabel_visible(False)
            dec.set_ticklabel_visible(False)
            from matplotlib import colors
            cmap = colors.ListedColormap([
                'xkcd:bright blue', 'xkcd:muted green', 'xkcd:goldenrod',
                'xkcd:light red'
            ])
            bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
            norm = colors.BoundaryNorm(bounds, cmap.N)

            pl.imshow(bpt_map[:, ::-1], cmap=cmap, vmin=-1, vmax=2)

            if colorbar == True:
                # pl.colorbar()
                cbar = pl.colorbar(ticks=[-1, 0, 1, 2])
                cbar.ax.set_yticklabels(['SF', 'Comp.', 'LIER', 'Sy.'])
            # pl.title("EDGE CO(1-0) map, W3 resolution")

        if return_map:
            return bpt_map

    def map_alpha_co(self,
                     galname,
                     which_alpha='fixed',
                     ax=None,
                     subplot=None,
                     return_map=False,
                     plot_map=False,
                     colorbar=True):
        '''
        Return and/or plot alpha_co
        '''

        accepted_alpha_co = ['fixed', 'met_alpha_sf_only', 'fixed_sf_only']
        if which_alpha not in accepted_alpha_co:
            print("*ERROR* alpha_co must be one of:")
            print(accepted_alpha_co)
            return None

        if ax is None:
            ax = pl

        alpha_co_fixed = 3.2

        if which_alpha == 'met_alpha_sf_only':
            alpha_co = pickle.load(
                open(reproj_c.fname_alpha_co_stacked(galname),
                     'rb'))['alpha_co']

        if which_alpha == 'fixed_sf_only':
            alpha_co_tmp = pickle.load(
                open(reproj_c.fname_alpha_co_stacked(galname),
                     'rb'))['alpha_co']
            alpha_co = np.ones(alpha_co_tmp.shape) * alpha_co_fixed
            alpha_co[np.isnan(alpha_co_tmp)] = np.nan

        if plot_map:
            # Plot the map
            mdat, mhdr = fits.getdata(
                '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
                % (galname, ),
                header=True)
            w = wcs.WCS(mhdr)
            idx = np.where(leda['Name'] == galname)[0][0]
            ractr = leda['ledaRA'].quantity[idx].to(u.deg).value
            dcctr = leda['ledaDE'].quantity[idx].to(u.deg).value

            x, y = w.wcs_world2pix(ractr, dcctr, 0)
            wcs_out = wcs.WCS(naxis=2)
            wcs_out.wcs.crpix = [x, y]
            # what is the galactic coordinate of that pixel.
            wcs_out.wcs.crval = [ractr, dcctr]
            # what is the pixel scale in lon, lat.
            wcs_out.wcs.cdelt = np.array([mhdr['CDELT1'], mhdr['CDELT2']])
            wcs_out.wcs.ctype = ['RA---SIN', 'DEC--SIN']
            if subplot is None:
                ax = pl.subplot(111, projection=wcs_out)
            else:
                ax = pl.subplot(subplot[0],
                                subplot[1],
                                subplot[2],
                                projection=wcs_out)
            ax = pl.gca()
            ra, dec = ax.coords[0], ax.coords[1]
            ra.set_ticklabel_visible(False)
            dec.set_ticklabel_visible(False)
            pl.imshow(alpha_co[:, ::-1], cmap='magma')

            if colorbar == True:
                pl.colorbar()
            # pl.title("EDGE CO(1-0) map, W3 resolution")

        if return_map:
            return alpha_co

    def map_sigma_h2(self,
                     galname,
                     which_alpha='fixed',
                     ax=None,
                     subplot=None,
                     return_map=False,
                     plot_map=False,
                     colorbar=True):
        '''
        Return and/or plot log Sigma H2 (Msun/pc^2)
        '''
        alpha_co_fixed = 3.2

        if ax is None:
            ax = pl

        mom0_map = gal_dict[galname]['LCO_map']

        if which_alpha != 'fixed':
            alpha_co = self.map_alpha_co(galname,
                                         which_alpha=which_alpha,
                                         return_map=True,
                                         plot_map=False)
            mom0_map /= alpha_co_fixed
            mom0_map *= alpha_co
            mom0_map = np.log10(mom0_map)
        else:
            mom0_map = np.log10(mom0_map)

        if plot_map:
            # Plot the map
            mdat, mhdr = fits.getdata(
                '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
                % (galname, ),
                header=True)
            w = wcs.WCS(mhdr)
            idx = np.where(leda['Name'] == galname)[0][0]
            ractr = leda['ledaRA'].quantity[idx].to(u.deg).value
            dcctr = leda['ledaDE'].quantity[idx].to(u.deg).value

            x, y = w.wcs_world2pix(ractr, dcctr, 0)
            wcs_out = wcs.WCS(naxis=2)
            wcs_out.wcs.crpix = [x, y]
            # what is the galactic coordinate of that pixel.
            wcs_out.wcs.crval = [ractr, dcctr]
            # what is the pixel scale in lon, lat.
            wcs_out.wcs.cdelt = np.array([mhdr['CDELT1'], mhdr['CDELT2']])
            wcs_out.wcs.ctype = ['RA---SIN', 'DEC--SIN']

            if subplot is None:
                ax = pl.subplot(111, projection=wcs_out)
            else:
                ax = pl.subplot(subplot[0],
                                subplot[1],
                                subplot[2],
                                projection=wcs_out)
            ax = pl.gca()
            ra, dec = ax.coords[0], ax.coords[1]
            ra.set_ticklabel_visible(False)
            dec.set_ticklabel_visible(False)
            im = pl.imshow(mom0_map[:, ::-1])
            if colorbar == True:
                pl.colorbar()

            # pl.title("EDGE CO(1-0) map, W3 resolution")

        if return_map:
            return mom0_map

    def map_sigma_h2_err(self,
                         galname,
                         which_alpha='fixed',
                         ax=None,
                         subplot=None,
                         return_map=False,
                         plot_map=False,
                         colorbar=True,
                         linear=False):
        '''
        Return and/or plot error in log Sigma H2 (Msun/pc^2)

        linear (bool) : if True, return error on Sigma H2. Otherwise
            return error on log Sigma H2.
        '''

        alpha_co_fixed = 3.2

        if ax is None:
            ax = pl

        mom0_map = gal_dict[galname]['LCO_map']
        mom0_map_err = gal_dict[galname]['LCO_map_err']

        if which_alpha != 'fixed':
            alpha_co = self.map_alpha_co(galname,
                                         which_alpha=which_alpha,
                                         return_map=True,
                                         plot_map=False)

            mom0_map_err /= alpha_co_fixed
            mom0_map_err *= alpha_co

            mom0_map /= alpha_co_fixed
            mom0_map *= alpha_co

        if not linear:
            mom0_map_err = 0.434 * mom0_map_err / mom0_map

        if plot_map:
            # Plot the map
            mdat, mhdr = fits.getdata(
                '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
                % (galname, ),
                header=True)
            w = wcs.WCS(mhdr)
            idx = np.where(leda['Name'] == galname)[0][0]
            ractr = leda['ledaRA'].quantity[idx].to(u.deg).value
            dcctr = leda['ledaDE'].quantity[idx].to(u.deg).value

            x, y = w.wcs_world2pix(ractr, dcctr, 0)
            wcs_out = wcs.WCS(naxis=2)
            wcs_out.wcs.crpix = [x, y]
            # what is the galactic coordinate of that pixel.
            wcs_out.wcs.crval = [ractr, dcctr]
            # what is the pixel scale in lon, lat.
            wcs_out.wcs.cdelt = np.array([mhdr['CDELT1'], mhdr['CDELT2']])
            wcs_out.wcs.ctype = ['RA---SIN', 'DEC--SIN']

            if subplot is None:
                ax = pl.subplot(111, projection=wcs_out)
            else:
                ax = pl.subplot(subplot[0],
                                subplot[1],
                                subplot[2],
                                projection=wcs_out)
            ax = pl.gca()
            ra, dec = ax.coords[0], ax.coords[1]
            ra.set_ticklabel_visible(False)
            dec.set_ticklabel_visible(False)
            im = pl.imshow(mom0_map_err[:, ::-1])
            if colorbar == True:
                pl.colorbar()

            # pl.title("EDGE CO(1-0) map, W3 resolution")

        if return_map:
            return mom0_map_err

    def map_sigma_12um(self,
                       galname,
                       ax=None,
                       subplot=None,
                       return_map=False,
                       plot_map=False,
                       colorbar=True):
        '''
        Return and/or plot log Sigma 12um (Lsun/pc^2)
        '''
        if ax is None:
            ax = pl

        image_w3_proj = np.log10(gal_dict[galname]['Lwise_map'])

        if plot_map:
            # Plot the map
            mdat, mhdr = fits.getdata(
                '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
                % (galname, ),
                header=True)
            w = wcs.WCS(mhdr)
            idx = np.where(leda['Name'] == galname)[0][0]
            ractr = leda['ledaRA'].quantity[idx].to(u.deg).value
            dcctr = leda['ledaDE'].quantity[idx].to(u.deg).value

            x, y = w.wcs_world2pix(ractr, dcctr, 0)
            wcs_out = wcs.WCS(naxis=2)
            wcs_out.wcs.crpix = [x, y]
            # what is the galactic coordinate of that pixel.
            wcs_out.wcs.crval = [ractr, dcctr]
            # what is the pixel scale in lon, lat.
            wcs_out.wcs.cdelt = np.array([mhdr['CDELT1'], mhdr['CDELT2']])
            wcs_out.wcs.ctype = ['RA---SIN', 'DEC--SIN']

            if subplot is None:
                ax = pl.subplot(111, projection=wcs_out)
            else:
                ax = pl.subplot(subplot[0],
                                subplot[1],
                                subplot[2],
                                projection=wcs_out)

            ra, dec = ax.coords[0], ax.coords[1]
            ra.set_ticklabel_visible(False)
            dec.set_ticklabel_visible(False)
            im = pl.imshow(image_w3_proj[:, ::-1], cmap='hot')
            if colorbar == True:
                pl.colorbar()

        if return_map:
            return image_w3_proj

    def map_sigma_12um_err(self,
                           galname,
                           ax=None,
                           subplot=None,
                           return_map=False,
                           plot_map=False,
                           colorbar=True,
                           linear=False):
        '''
        Return and/or plot error in log Sigma 12um (Lsun/pc^2)
        '''
        if ax is None:
            ax = pl

        image_w3_proj = np.log10(gal_dict[galname]['Lwise_map'])
        if linear == False:
            image_w3_proj_err = 0.434 * gal_dict[galname][
                'Lwise_map_err'] / gal_dict[galname]['Lwise_map']
        else:
            image_w3_proj_err = gal_dict[galname]['Lwise_map_err']

        if plot_map:
            # Plot the map
            mdat, mhdr = fits.getdata(
                '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
                % (galname, ),
                header=True)
            w = wcs.WCS(mhdr)
            idx = np.where(leda['Name'] == galname)[0][0]
            ractr = leda['ledaRA'].quantity[idx].to(u.deg).value
            dcctr = leda['ledaDE'].quantity[idx].to(u.deg).value

            x, y = w.wcs_world2pix(ractr, dcctr, 0)
            wcs_out = wcs.WCS(naxis=2)
            wcs_out.wcs.crpix = [x, y]
            # what is the galactic coordinate of that pixel.
            wcs_out.wcs.crval = [ractr, dcctr]
            # what is the pixel scale in lon, lat.
            wcs_out.wcs.cdelt = np.array([mhdr['CDELT1'], mhdr['CDELT2']])
            wcs_out.wcs.ctype = ['RA---SIN', 'DEC--SIN']

            if subplot is None:
                ax = pl.subplot(111, projection=wcs_out)
            else:
                ax = pl.subplot(subplot[0],
                                subplot[1],
                                subplot[2],
                                projection=wcs_out)

            ra, dec = ax.coords[0], ax.coords[1]
            ra.set_ticklabel_visible(False)
            dec.set_ticklabel_visible(False)
            im = pl.imshow(image_w3_proj_err[:, ::-1], cmap='hot')
            if colorbar == True:
                pl.colorbar()

        if return_map:
            return image_w3_proj_err

    def plot_sigma_h2_vs_sigma_12(self,
                                  galname,
                                  which_alpha='fixed',
                                  which_fit='linmix_fit_reverse',
                                  ax=None,
                                  subplot=None):
        '''
        Plot log H2 surface density vs log 12 um surface density, along with the fit, if available.

        Args:
            which_alpha (str) : Which conversion factor to use? Currently accepted
                are...
                    1. "fixed", i.e. alpha_co = 3.2
                    2. "met_alpha_sf_only", i.e. metallicity-dependent, and only
                        covering star-forming pixels.
                    3. "fixed_sf_only", i.e. alpha_co = 3.2 but only over star-forming
                        pixels.
            which_fit (str) : Which fit to plot? Note that "reverse" means
                y = log Sigma H2, and x = log Sigma 12um, while "forward" means the
                opposite. Currently accepted are...
                    1. "linmix_fit_reverse" / "linmix_fit_forward", i.e. fits
                        using LinMix over all pixels (assuming "fixed" alpha_co).
                    2.  "fit_result_reverse" / "fit_result_forward", same as above but
                        for the other two alpha_co scenarios.
                    3.  "lts_fit_reverse" / "lts_fit_forward", using LTS fitting
                        (Cappellari), assuming "fixed" alpha_co. This also applies
                        to the other two alpha_co scenarios.
        '''
        if len(
                glob.glob(
                    '/Users/ryan/venus/shared_data/califa/DR3-stack/%s/%s_result.pk'
                    % (galname, galname))) == 0:
            print("Nothing exists for this galaxy")
            return

        gal_dict_i = gal_dict[galname]
        pixel_area_pc2 = surf_dens_all[galname]['pix_area_pc2']

        if which_fit.split('_')[0] == 'linmix':
            check_type = dict
        if which_fit.split('_')[0] == 'lts':
            check_type = lts_linefit
        if which_fit.split('_')[0] == 'fit':
            check_type = dict

        # Which alpha_co do you want?
        if which_alpha == 'fixed':
            fit_i = gal_dict_i
            x_fit = gal_dict[galname]['x_fit']
            x_err_fit = gal_dict[galname]['x_err_fit']
            y_fit = gal_dict[galname]['y_fit']
            y_err_fit = gal_dict[galname]['y_err_fit']
        if which_alpha == 'fixed_sf_only':
            fit_i = gal_dict_i['fits_fixed_alpha_sf_only']
            x_fit = gal_dict[galname]['x_fit_sf']
            x_err_fit = gal_dict[galname]['x_fit_sf_err']
            y_fit = gal_dict[galname]['y_fit_sf']
            y_err_fit = gal_dict[galname]['y_fit_sf_err']
        if which_alpha == 'met_alpha_sf_only':
            fit_i = gal_dict_i['result_met_alpha_sf_only']
            x_fit = gal_dict[galname]['x_fit_met']
            x_err_fit = gal_dict[galname]['x_fit_met_err']
            y_fit = gal_dict[galname]['y_fit_met']
            y_err_fit = gal_dict[galname]['y_fit_met_err']

        if 'reverse' in which_fit:
            x_fit, x_err_fit, y_fit, y_err_fit = y_fit, y_err_fit, x_fit, x_err_fit

        if (type(fit_i[which_fit]) == check_type):
            if check_type == dict:
                slope_best = np.median(fit_i[which_fit]['chains'][1]) #fit_i[which_fit]['slope']
                intercept_best = np.median(fit_i[which_fit]['chains'][0]) #fit_i[which_fit]['intercept']
                slope_err = fit_i[which_fit]['slope_err']
                intercept_err = fit_i[which_fit]['intercept_err']
                if which_fit.split('_')[-1] == 'masked':
                    if np.where(fit_i['lts_fit_mask'] == True)[0].size < 6:
                        return
            else:
                intercept_best, slope_best = fit_i[which_fit].ab
                intercept_err, slope_err = fit_i[which_fit].ab_err

            if ax is None:
                pl.figure(figsize=(8, 4))
                ax = pl

            if 'lts' in which_fit:
                fit_i[which_fit].plot(x_fit, y_fit, x_err_fit, y_err_fit)
            else:
                ax.errorbar(x_fit,
                            y_fit,
                            xerr=x_err_fit,
                            yerr=y_err_fit,
                            marker='s',
                            color='k',
                            markeredgecolor='k',
                            markerfacecolor='xkcd:goldenrod',
                            alpha=1,
                            capsize=2,
                            markersize=5,
                            ecolor='k',
                            elinewidth=0.5,
                            markeredgewidth=0.5,
                            linestyle='none')

                xmin, xmax = np.min(x_fit), np.max(x_fit)
                dx = (xmax - xmin) / 2.
                xlimits = np.array([xmin - dx, xmax + dx])

                plot_fit(xlimits,
                         intercept_best,
                         slope_best,
                         a_err=intercept_err,
                         b_err=slope_err,
                         xin=x_fit,
                         yin=y_fit,
                         s=None,
                         pivot=0,
                         ax=None,
                         log=False,
                         color='xkcd:bright blue',
                         lw=2,
                         alpha=0.5)
            return
        else:
            ax.errorbar(x_fit,
                        y_fit,
                        xerr=x_err_fit,
                        yerr=y_err_fit,
                        marker='s',
                        color='k',
                        markeredgecolor='k',
                        markerfacecolor='xkcd:goldenrod',
                        alpha=1,
                        capsize=2,
                        markersize=5,
                        ecolor='k',
                        elinewidth=0.5,
                        markeredgewidth=0.5,
                        linestyle='none')

    def slope_intercept(self,
                        which_fit,
                        which_alpha,
                        galname=None,
                        galname_list=None,
                        return_fits=False,
                        plot_fits=False,
                        perform_fit=False,
                        perform_which_fit=None,
                        markerfacecolor='xkcd:goldenrod',
                        label=None,
                        marker='s'):
        '''
        Get the best fit slope(s) and intercept(s) (and their errors)
        for a galaxy (or list of galaxies).
        Must specify which fit you want, and which alpha_co
        (see "plot_sigma_h2_vs_sigma_12" for valid alphas and fits).
        '''
        if (galname is None) and (galname_list is None):
            print(
                "Need to provide at least one galaxy name or a list of names.")
            return

        if (galname is not None) and (galname_list is not None):
            print("Provide either one name or a list of names.")
            return

        res = dict()
        res['name'] = []
        res['slope'] = []
        res['slope_err'] = []
        res['intercept'] = []
        res['intercept_err'] = []

        if galname is not None:
            galname_list = [galname]

        if which_fit.split('_')[0] == 'linmix':
            check_type = dict
        if which_fit.split('_')[0] == 'lts':
            check_type = lts_linefit
        if which_fit.split('_')[0] == 'fit':
            check_type = dict

        for galname in galname_list:
            gal_dict_i = gal_dict[galname]
            pixel_area_pc2 = surf_dens_all[galname]['pix_area_pc2']

            # Which alpha_co do you want?
            if which_alpha == 'fixed':
                fit_i = gal_dict_i
                x_fit = gal_dict[galname]['x_fit']
                x_err_fit = gal_dict[galname]['x_err_fit']
                y_fit = gal_dict[galname]['y_fit']
                y_err_fit = gal_dict[galname]['y_err_fit']
            if which_alpha == 'fixed_sf_only':
                fit_i = gal_dict_i['fits_fixed_alpha_sf_only']
                x_fit = gal_dict[galname]['x_fit_sf']
                x_err_fit = gal_dict[galname]['x_fit_sf_err']
                y_fit = gal_dict[galname]['y_fit_sf']
                y_err_fit = gal_dict[galname]['y_fit_sf_err']
            if which_alpha == 'met_alpha_sf_only':
                fit_i = gal_dict_i['result_met_alpha_sf_only']
                x_fit = gal_dict[galname]['x_fit_met']
                x_err_fit = gal_dict[galname]['x_fit_met_err']
                y_fit = gal_dict[galname]['y_fit_met']
                y_err_fit = gal_dict[galname]['y_fit_met_err']

            if 'reverse' in which_fit:
                x_fit, x_err_fit, y_fit, y_err_fit = y_fit, y_err_fit, x_fit, x_err_fit

            if (type(fit_i[which_fit]) == check_type):
                if check_type == dict:
                    slope_best = np.median(fit_i[which_fit]['chains'][1]) #fit_i[which_fit]['slope']
                    intercept_best = np.median(fit_i[which_fit]['chains'][0]) #fit_i[which_fit]['intercept']
                    # slope_best = fit_i[which_fit]['slope']
                    # intercept_best = fit_i[which_fit]['intercept']
                    slope_err = fit_i[which_fit]['slope_err']
                    intercept_err = fit_i[which_fit]['intercept_err']
                    if which_fit.split('_')[-1] == 'masked':
                        if np.where(fit_i['lts_fit_mask'] == True)[0].size < 6:
                            return
                else:
                    intercept_best, slope_best = fit_i[which_fit].ab
                    intercept_err, slope_err = fit_i[which_fit].ab_err

                res['name'].append(galname)
                res['slope'].append(slope_best)
                res['slope_err'].append(slope_err)
                res['intercept'].append(intercept_best)
                res['intercept_err'].append(intercept_err)

        res['name'] = np.array(res['name'])
        res['slope'] = np.array(res['slope'])
        res['slope_err'] = np.array(res['slope_err'])
        res['intercept'] = np.array(res['intercept'])
        res['intercept_err'] = np.array(res['intercept_err'])

        if plot_fits == True:
            # Plot the fits
            if perform_fit == False:
                pl.errorbar(res['intercept'],
                            res['slope'],
                            xerr=res['intercept_err'],
                            yerr=res['slope_err'],
                            marker=marker,
                            color='k',
                            markeredgecolor='k',
                            markerfacecolor=markerfacecolor,
                            alpha=1,
                            capsize=2,
                            markersize=5,
                            ecolor='k',
                            elinewidth=0.5,
                            markeredgewidth=0.5,
                            linestyle='none',
                            label=label)
            else:
                if perform_which_fit is None:
                    print("Doing default fit: linmix")
                    perform_which_fit = 'linmix'
                if perform_which_fit == 'lts':
                    lts_linefit(res['intercept'], res['slope'],
                                res['intercept_err'], res['slope_err'])
                if perform_which_fit == 'linmix':
                    pl.errorbar(res['intercept'],
                                res['slope'],
                                xerr=res['intercept_err'],
                                yerr=res['slope_err'],
                                marker=marker,
                                color='k',
                                markeredgecolor='k',
                                markerfacecolor=markerfacecolor,
                                alpha=1,
                                capsize=2,
                                markersize=5,
                                ecolor='k',
                                elinewidth=0.5,
                                markeredgewidth=0.5,
                                linestyle='none',
                                label=label)
                    x_fit, y_fit, x_err, y_err = res['intercept'], res[
                        'slope'], res['intercept_err'], res['slope_err']
                    fit = run_linmix(x_fit, y_fit, x_err, y_err)
                    xmin, xmax = np.min(x_fit), np.max(x_fit)
                    dx = (xmax - xmin) / 2.
                    xlimits = np.array([xmin - dx, xmax + dx])
                    plot_fit(xlimits,
                             fit['intercept'],
                             fit['slope'],
                             a_err=fit['intercept_err'],
                             b_err=fit['slope_err'],
                             xin=x_fit,
                             yin=y_fit,
                             s=None,
                             pivot=0,
                             ax=None,
                             log=False,
                             color='xkcd:bright blue',
                             lw=2,
                             alpha=0.5)
        if return_fits == True:
            return res


def get_inter_califa(galname):
    '''
    Get interacting / isolated status from CALIFA data tables
    '''
    f_ms = fits.open(
        '/Users/ryan/Dropbox/mac/tsinghua/CALIFA_2_MS_class.fits.txt')
    califa_ms_names = f_ms[1].data['REALNAME']
    if galname in califa_ms_names:
        return f_ms[1].data['merg'][califa_ms_names == galname]
    else:
        f_es = fits.open('/Users/ryan/venus/home/CALIFA_2_ES_class.fits')
        califa_es_names = f_es[1].data['realname']
        return f_es[1].data['merg'][califa_es_names == galname]


def get_hubble_type_califa(galname):
    '''
    Get hubble type from CALIFA data tables
    '''
    f_ms = fits.open(
        '/Users/ryan/Dropbox/mac/tsinghua/CALIFA_2_MS_class.fits.txt')
    califa_ms_names = f_ms[1].data['REALNAME']
    if galname in califa_ms_names:
        htype = f_ms[1].data['hubtyp'][califa_ms_names == galname]
        htype_min = f_ms[1].data['minhubtyp'][califa_ms_names == galname]
        htype_max = f_ms[1].data['maxhubtyp'][califa_ms_names == galname]
        return htype[0], htype_min[0], htype_max[0]
    else:
        f_es = fits.open('/Users/ryan/venus/home/CALIFA_2_ES_class.fits')
        califa_es_names = f_es[1].data['realname']
        htype = f_es[1].data['hubtyp'][califa_es_names == galname]
        htype_min = f_es[1].data['minhubtyp'][califa_es_names == galname]
        htype_max = f_es[1].data['maxhubtyp'][califa_es_names == galname]
        return htype[0], htype_min[0], htype_max[0]


def run_linmix(x, y, xerr, yerr, parallelize=False, nchains=4):
    lm_result = linmix.LinMix(x,
                              y,
                              xerr,
                              yerr,
                              K=3,
                              parallelize=parallelize,
                              nchains=nchains)
    lm_result.run_mcmc(silent=True)
    chains = np.vstack([lm_result.chain['alpha'], lm_result.chain['beta']])
    result = dict()
    result['chains'] = chains
    result['intercept'] = np.average(chains[0])
    result['intercept_err'] = np.std(chains[0])
    result['slope'] = np.average(chains[1])
    result['slope_err'] = np.std(chains[1])
    return result


def scatterplot(x,
                y,
                xlabel,
                ylabel,
                xerr=None,
                yerr=None,
                ax=None,
                label=None,
                fontsize_axes=10,
                **kwargs):
    marker = 'o'
    mec = 'k'
    c = 'b'
    ecolor = None
    if 'marker' in kwargs:
        # print("Changing marker")
        marker = kwargs['marker']
    if 'markeredgecolor' in kwargs:
        mec = kwargs['markeredgecolor']
    if 'color' in kwargs:
        c = kwargs['color']
    if 'ecolor' in kwargs:
        ecolor = kwargs['ecolor']

    if (xerr is not None) or (yerr is not None):
        pl.errorbar(x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    marker=marker,
                    markersize=4.,
                    markerfacecolor='none',
                    markeredgecolor=mec,
                    markeredgewidth=0.5,
                    capsize=2,
                    linestyle='none',
                    label=label,
                    elinewidth=.5,
                    ecolor=ecolor)
    else:
        pl.scatter(x,
                   y,
                   marker=marker,
                   s=5.,
                   facecolor='none',
                   linewidth=0.5,
                   linestyle='None',
                   color=c,
                   label=label)
    if xlabel != '':
        pl.xlabel(xlabel, fontsize=fontsize_axes)
    if ylabel != '':
        pl.ylabel(ylabel, fontsize=fontsize_axes)


fname_combos = '/Users/ryan/Dropbox/mac/wise_w3_vs_co/combos.pk'

# New file I found on Tony Wong's github -- note there are different versions with different things
edge_co_params = pd.read_csv(
    '/Users/ryan/Dropbox/mac/tsinghua/edge_pydb-master/dat_glob/derived/build/EDGE_COparameters.csv',
    skiprows=18)
#use it for offsets of the images and inclinations
names_ecopars, xoffs, yoffs, inc_ecopars = np.array(
    edge_co_params["Name"]), np.array(edge_co_params[" xoff"]), np.array(
        edge_co_params[" yoff"]), np.array(edge_co_params[' Inc'])

cosmo = FlatLambdaCDM(H0=70.2, Om0=0.275, Tcmb0=2.725)
redshifts = np.loadtxt(
    '/Users/ryan/Downloads/stz349_supplemental_files/TableA1.csv',
    usecols=5,
    delimiter=',',
    skiprows=1)
names = np.loadtxt(
    '/Users/ryan/Downloads/stz349_supplemental_files/TableA1.csv',
    usecols=0,
    delimiter=',',
    skiprows=1,
    dtype=str)
bam = np.loadtxt('/Users/ryan/Downloads/stz349_supplemental_files/TableA1.csv',
                 usecols=1,
                 delimiter=',',
                 skiprows=1,
                 dtype=str)

# fnames_mom0 = glob.glob('/Users/ryan/venus/shared_data/edge/moment0_W3_26April19/*mom0.fits')
fnames_mom0 = glob.glob(
    '/Users/ryan/venus/shared_data/edge/moment0_W3_30April19/*mom0.fits')

# edge_list = glob.glob('/Users/ryan/venus/shared_data/edge/signal_cubes/*.co.cmmsk.fits')
# galnames = [ nm.split('/')[-1].split('.')[0] for nm in edge_list ]
edge_list = glob.glob('/Users/ryan/venus/shared_data/edge/W3/image_half/*')
galnames = [nm.split('/')[-1] for nm in edge_list]

califa_basic = fits.open('/Users/ryan/Dropbox/mac/CALIFA_1_MS_basic.fits')
califa_basic_names = califa_basic[1].data.field('REALNAME')
califa_basic_z = califa_basic[1].data.field('redshift')

califa_es_basic = fits.open('/Users/ryan/Dropbox/mac/CALIFA_1_ES_basic.fits')
califa_es_names = califa_es_basic[1].data.field('REALNAME')
califa_es_z = califa_es_basic[1].data.field('redshift')

leda = Table.read(
    '/usr/local/lib/python3.7/site-packages/edge_pydb/dat_glob/external/edge_leda.csv',
    format='ascii.ecsv')

# PA and INC from Becca's table
rfpars = Table.read(
    '/Users/ryan/Dropbox/mac/tsinghua/edge_pydb-master/dat_glob/derived/edge_rfpars.csv',
    format='ascii.ecsv')

edge_califa = Table.read(
    '/usr/local/lib/python3.7/site-packages/edge_pydb/dat_glob/external/edge_califa.csv',
    format='ascii.ecsv')

# fname_l12_lco_dict = '/Users/ryan/Dropbox/mac/wise_w3_vs_co/l12_lco_dict_v4.pk'  # Without Aniano kernels, until that is fixed, and with more conservative s/n cuts in mom0 map making
# fname_l12_lco_dict = '/Users/ryan/Dropbox/mac/wise_w3_vs_co/l12_lco_dict_v5.pk'  # Without Aniano kernels, until that is fixed, and with more conservative s/n cuts in mom0 map making
fname_l12_lco_dict = '/Users/ryan/Dropbox/mac/wise_w3_vs_co/l12_lco_dict_withfits_v5.pk'
gal_dict = pickle.load(open(fname_l12_lco_dict, 'rb'))

Lsun = 3.839e33  # Lsun in erg/s

fname_mom0_jypb_sun_rebinned = lambda galname: '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_sun_rebin6_jypb_mom0.fits' % (
    galname, )
fname_noise_map = lambda galname: '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_noise_jypbkms.fits' % (
    galname, )
fname_noise_map_total = lambda galname: '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_total_noise_rebin6_jypbkms.fits' % (
    galname, )


# Make 3x3 templates for any number of connected pixels
def generate_templates_central1():
    """
    Generate all possible 3x3 matrices
    with ones in the center, and n ones around it, where
    n = 1, 2, 3, ..., 8
    """

    base = np.zeros((3, 3), dtype=int)
    base[1, 1] = 1
    combos = dict()
    combos['1'] = base

    combos_n = []
    for i in range(0, 3):
        for j in range(0, 3):
            template_i = base.copy()
            template_i[i, j] = 1
            if template_i.sum() == 2:
                combos_n.append(template_i)
    combos_n = np.unique(np.array(combos_n), axis=0)
    combos['2'] = combos_n

    for n in range(3, 10):
        print("n = %i" % (n))
        combos_n = []
        for base_i in combos[str(n - 1)]:
            for i in range(0, 3):
                for j in range(0, 3):
                    template_i = base_i.copy()
                    template_i[i, j] = 1
                    if template_i.sum() == n:
                        combos_n.append(template_i)
        combos_n = np.unique(np.array(combos_n), axis=0)
        combos[str(n)] = combos_n
    return combos


def generate_templates_all():
    """
    Generate all possible 3x3 matrices with n ones that are *connected*, where
    n = 1, 2, 3, ..., 9.

    """
    combos = dict()

    combos_n = []
    for i in range(0, 3):
        for j in range(0, 3):
            template_i = np.zeros((3, 3), dtype=int)
            template_i[i, j] = 1
            if template_i.sum() == 1:
                combos_n.append(template_i)
    combos_n = np.unique(np.array(combos_n), axis=0)
    combos['1'] = combos_n

    combos_n = []
    for base in combos['1']:
        for i in range(0, 3):
            for j in range(0, 3):
                template_i = base.copy()
                template_i[i, j] = 1
                if template_i.sum() == 2:
                    combos_n.append(template_i)
    combos_n = np.unique(np.array(combos_n), axis=0)
    combos['2'] = combos_n

    for n in range(3, 10):
        print("n = %i" % (n))
        combos_n = []
        for base_i in combos[str(n - 1)]:
            for i in range(0, 3):
                for j in range(0, 3):
                    template_i = base_i.copy()
                    template_i[i, j] = 1
                    if template_i.sum() == n:
                        combos_n.append(template_i)
        combos_n = np.unique(np.array(combos_n), axis=0)
        combos[str(n)] = combos_n
    return combos


def compute_surface_densities_combos_12um_and_co(other_maps=None,
                                                 other_maps_names=None,
                                                 other_maps_err=None):
    """
    Use the 3x3 matrices from 'generate_templates_all'
    to compute surface densities of 12um and CO over
    areas of n=1,...,9 pixels that are *connected*.

    For a given n, each detected pixel is only used once. So
    every resulting surface density at a given n is independent.

    Args:
        other_maps :

        other_maps_names :

        other_maps_err (list) : True if error, False otherwise

    Returns:
        surf_dens_all (dict) : Dictionary containing all surface densities
            and corresponding uncertainties for n=1,...,9, for all galaxies.
            Note that the surface densities for each n are in linear units.
            The global values and their uncertainties are in log units. 'n_max'
            is the maximum n for a given galaxy.
    """

    combos = pickle.load(open(fname_combos, 'rb'))

    surf_dens_all = dict()

    for galname in galnames:
        gal = galname
        if galname[:3] == 'UGC':
            if len(
                    glob.glob(
                        '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
                        % (galname, ))) == 0:
                gal = 'UGC0' + galname[3:]
            if len(
                    glob.glob(
                        '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
                        % (gal, ))) == 0:
                gal = 'UGC00' + galname[3:]
        if galname[:2] == 'IC':
            if len(
                    glob.glob(
                        '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
                        % (galname, ))) == 0:
                gal = 'IC0' + galname[2:]
        if galname[:3] == 'NGC':
            if len(
                    glob.glob(
                        '/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s_co_smooth_wise_v2_rebin6_mom0.fits'
                        % (galname, ))) == 0:
                gal = 'NGC0' + galname[3:]
        if (gal in ['UGC05359', 'UGC03253']) or (gal not in gal_dict.keys()):
            continue
        else:
            gal_dict_i = gal_dict[gal]

            pixel_size_arcsec = 6.
            pixel_area_pc2 = (pixel_size_arcsec * gal_dict_i['dist_Mpc'] *
                              1e6 / 206265.)**2

            # Convert x and y from log10 to linear units
            x = (10**gal_dict_i['x_fit'])
            x_err = gal_dict_i['x_err_fit'] * x / 0.434
            y = (10**gal_dict_i['y_fit'])
            y_err = gal_dict_i['y_err_fit'] * y / 0.434

            # Convert units from luminosity to surface density
            # x /= pixel_area_pc2
            # x_err /= pixel_area_pc2
            # y /= pixel_area_pc2
            # y_err /= pixel_area_pc2

            # Linear fit on x and y in units of log10(surface density)
            fit_flag = 1
            if x.size > 6:
                # Convert slope and intercept (from fits on log10(luminosity))
                # into units of log10(surface density)
                slope_best = gal_dict_i['linmix_fit_reverse']['slope']
                intercept_best = gal_dict_i['linmix_fit_reverse']['intercept']
                # intercept_best = intercept_best - slope_best * (
                #     -np.log10(pixel_area_pc2)) - np.log10(pixel_area_pc2)
            else:
                fit_flag = 0
            xmin, xmax = -1, 2
            ymin, ymax = -0.5, 0.5

            mask_arr = np.zeros(gal_dict_i['LCO_map'].shape, dtype=int)
            mask_arr[np.isnan(gal_dict_i['LCO_map']) == False] = 1

            if mask_arr.sum() == 0:
                continue

            # "sigma" means surface density
            sigma_arr_co = gal_dict_i['LCO_map'] * mask_arr  # / pixel_area_pc2
            sigma_arr_co[np.isnan(gal_dict_i['LCO_map'])] = 0

            sigma_arr_co_err = gal_dict_i[
                'LCO_map_err'] * mask_arr  #/ pixel_area_pc2
            sigma_arr_co_err[np.isnan(gal_dict_i['LCO_map_err'])] = 0

            sigma_arr_l12 = gal_dict_i[
                'Lwise_map'] * mask_arr  # / pixel_area_pc2
            sigma_arr_l12[np.isnan(gal_dict_i['LCO_map'])] = 0

            sigma_arr_l12_err = gal_dict_i[
                'Lwise_map_err'] * mask_arr  #/ pixel_area_pc2
            sigma_arr_l12_err[np.isnan(gal_dict_i['LCO_map'])] = 0

            # Load any other maps if provided
            do_other_maps = False
            if (other_maps is not None) and (other_maps_names is not None):
                do_other_maps = True
                n_other_maps = len(other_maps)

                sigma_arrays_other = dict()
                sigmas_other = dict()
                error_other = dict()

                i = 0
                for map_name in other_maps_names:
                    fun_get_other_map_i = other_maps[i]
                    # Each one of these will be like 'sigmas_co', 'sigmas_l12' etc. below
                    sigmas_other[map_name] = dict()
                    # Each one of these is like 'sigma_arr_co', etc. above
                    # # extra_factor = np.cos(np.radians(gal_dict_i['incl']))
                    # if (map_name == 'halpha') or (map_name == 'halpha_err'):
                    #     d_cm = np.float64(
                    #         (gal_dict_i['dist_Mpc'] * u.Mpc).to(u.cm).value)
                    #     extra_factor = 4. * np.pi * d_cm**2 * 5.3e-42 * 1e-16 * 1e6
                    try:
                        sigma_arrays_other[map_name] = fun_get_other_map_i(
                            gal)  # * extra_factor / pixel_area_pc2
                    except:
                        sigma_arrays_other[map_name] = np.ones(
                            sigma_arr_l12_err.shape) * np.nan

                    if type(sigma_arrays_other[map_name]) == int:
                        sigma_arrays_other[map_name] = np.ones(
                            sigma_arr_l12_err.shape) * np.nan
                    # Is this map an uncertainty?
                    error_other[map_name] = other_maps_err[i]
                    i += 1

            index_arr = np.arange(mask_arr.size)
            index_arr = index_arr.reshape(mask_arr.shape)

            i_det, j_det = np.nonzero(mask_arr)

            n_max = 1

            sigmas_co = dict()
            sigmas_l12 = dict()
            sigmas_co_err = dict()
            sigmas_l12_err = dict()
            indices = dict()
            masks = dict()
            counts = dict()

            # Loop over detected pixels
            for pixel_num in range(0, i_det.size):
                i = i_det[pixel_num]
                j = j_det[pixel_num]
                if (i == mask_arr.shape[0] - 1) or (j == mask_arr.shape[1] -
                                                    1):
                    continue
                # Cut out 3x3 centered at pixel
                mask_p = mask_arr[i - 1:i + 2, j - 1:j + 2]
                indices_p = index_arr[i - 1:i + 2, j - 1:j + 2]
                sigma_p_co = sigma_arr_co[i - 1:i + 2, j - 1:j + 2]
                sigma_p_l12 = sigma_arr_l12[i - 1:i + 2, j - 1:j + 2]

                sigma_p_co_err = sigma_arr_co_err[i - 1:i + 2, j - 1:j + 2]
                sigma_p_l12_err = sigma_arr_l12_err[i - 1:i + 2, j - 1:j + 2]

                if do_other_maps == True:
                    cutouts_other = dict()
                    for k in list(sigma_arrays_other.keys()):
                        cutouts_other[k] = sigma_arrays_other[k][i - 1:i +
                                                                 2, j - 1:j +
                                                                 2]

                # How many detected pixels in cutout?
                n_det_p = mask_p.sum()
                n_max = max(n_max, n_det_p)

                for n in range(1, n_det_p + 1):
                    if str(n) not in list(sigmas_co.keys()):
                        sigmas_co[str(n)] = []
                        sigmas_l12[str(n)] = []
                        sigmas_co_err[str(n)] = []
                        sigmas_l12_err[str(n)] = []
                        indices[str(n)] = []
                        masks[str(n)] = mask_arr.copy()
                        counts[str(n)] = np.zeros(mask_arr.shape)

                        if do_other_maps == True:
                            for k in list(sigma_arrays_other.keys()):
                                sigmas_other[k][str(n)] = []

                    # Try all templates
                    if n == 1:
                        template_i = np.zeros((3, 3), dtype=int)
                        template_i[1, 1] = 1
                        # Store stuff
                        sigmas_co[str(n)].append(
                            np.sum(sigma_p_co * template_i) / np.float(n))
                        sigmas_l12[str(n)].append(
                            np.sum(sigma_p_l12 * template_i) / np.float(n))
                        sigmas_co_err[str(n)].append(
                            np.sqrt(np.sum(sigma_p_co_err**2 * template_i)) /
                            np.float(n))
                        sigmas_l12_err[str(n)].append(
                            np.sqrt(np.sum(sigma_p_l12_err**2 * template_i)) /
                            np.float(n))
                        masked_indices = indices_p * template_i
                        indices[str(n)].append(
                            np.sort(
                                masked_indices[np.nonzero(masked_indices)]))

                        if do_other_maps == True:
                            for k in list(sigma_arrays_other.keys()):
                                if error_other[k] == True:
                                    sigmas_other[k][str(n)].append(
                                        np.sqrt(
                                            np.sum(cutouts_other[k]**2 *
                                                   template_i)) / np.float(n))
                                else:
                                    sigmas_other[k][str(n)].append(
                                        np.sum(cutouts_other[k] * template_i) /
                                        np.float(n))

                    else:
                        to_mask = mask_p.copy()
                        for template_i in combos[str(n)]:
                            mask_p_n = masks[str(n)][i - 1:i + 2, j - 1:j + 2]
                            counts_p_n = counts[str(n)][i - 1:i + 2, j - 1:j +
                                                        2]
                            if (mask_p_n * template_i).sum() == n:
                                # Store stuff
                                sigmas_co[str(n)].append(
                                    np.sum(sigma_p_co * template_i) /
                                    np.float(n))
                                sigmas_l12[str(n)].append(
                                    np.sum(sigma_p_l12 * template_i) /
                                    np.float(n))
                                sigmas_co_err[str(n)].append(
                                    np.sqrt(
                                        np.sum(sigma_p_co_err**2 * template_i))
                                    / np.float(n))
                                sigmas_l12_err[str(n)].append(
                                    np.sqrt(
                                        np.sum(sigma_p_l12_err**2 *
                                               template_i)) / np.float(n))
                                masked_indices = indices_p * template_i
                                indices[str(n)].append(
                                    np.sort(masked_indices[np.nonzero(
                                        masked_indices)]))

                                if do_other_maps == True:
                                    for k in list(sigma_arrays_other.keys()):
                                        if error_other[k] == True:
                                            sigmas_other[k][str(n)].append(
                                                np.sqrt(
                                                    np.sum(cutouts_other[k]**2
                                                           * template_i)) /
                                                np.float(n))
                                        else:
                                            sigmas_other[k][str(n)].append(
                                                np.sum(cutouts_other[k] *
                                                       template_i) /
                                                np.float(n))

                                counts[str(n)][i - 1:i + 2, j - 1:j +
                                               2] += template_i
                                masks[str(n)][counts[str(n)] == 1] = 0

            sigma_n_co = np.array(sigmas_co['1']).flatten()
            sigmas_co['1'] = sigma_n_co[np.nonzero(sigma_n_co)]
            sigmas_co_err['1'] = np.array(
                sigmas_co_err['1']).flatten()[np.nonzero(sigma_n_co)]
            sigma_n = np.array(sigmas_l12['1']).flatten()
            sigmas_l12['1'] = sigma_n[np.nonzero(sigma_n_co)]
            sigmas_l12_err['1'] = np.array(
                sigmas_l12_err['1']).flatten()[np.nonzero(sigma_n_co)]

            if do_other_maps == True:
                for k in list(sigma_arrays_other.keys()):
                    sigma_n_other = np.array(sigmas_other[k][str(1)]).flatten()
                    sigmas_other[k][str(1)] = sigma_n_other[np.nonzero(
                        sigma_n_co)]

            # k = 2
            for n in range(2, n_max + 1):
                if str(n) in list(sigmas_co.keys()):
                    indices_n = np.array(indices[str(n)])

                    sigma_n_co = np.array(sigmas_co[str(n)])
                    sigma_n_co = sigma_n_co[np.unique(
                        indices_n, axis=0, return_index=True)[1]].flatten()

                    sigma_n_co_err = np.array(sigmas_co_err[str(n)])
                    sigma_n_co_err = sigma_n_co_err[np.unique(
                        indices_n, axis=0, return_index=True)[1]].flatten()

                    sigmas_co[str(n)] = sigma_n_co[np.nonzero(sigma_n_co)]
                    sigmas_co_err[str(n)] = sigma_n_co_err[np.nonzero(
                        sigma_n_co)]

                    sigma_n = np.array(sigmas_l12[str(n)])
                    sigma_n = sigma_n[np.unique(
                        indices_n, axis=0, return_index=True)[1]].flatten()

                    sigma_n_err = np.array(sigmas_l12_err[str(n)])
                    sigma_n_err = sigma_n_err[np.unique(
                        indices_n, axis=0, return_index=True)[1]].flatten()

                    sigmas_l12[str(n)] = sigma_n[np.nonzero(sigma_n_co)]
                    sigmas_l12_err[str(n)] = sigma_n_err[np.nonzero(
                        sigma_n_co)]

                    if do_other_maps == True:
                        for k in list(sigma_arrays_other.keys()):
                            sigma_n_other = np.array(
                                sigmas_other[k][str(n)]).flatten()
                            sigma_n_other = sigma_n_other[np.unique(
                                indices_n, axis=0,
                                return_index=True)[1]].flatten()

                            sigmas_other[k][str(n)] = sigma_n_other[np.nonzero(
                                sigma_n_co)]

            npix = np.array([int(i) for i in list(sigmas_co.keys())])

            surf_dens_all[gal] = dict()
            surf_dens_all[gal]['pix_area_pc2'] = pixel_area_pc2
            surf_dens_all[gal]['npix_globals'] = y.size
            surf_dens_all[gal]['sigmas_co'] = sigmas_co
            surf_dens_all[gal]['sigmas_co_err'] = sigmas_co_err
            surf_dens_all[gal]['sigmas_l12'] = sigmas_l12
            surf_dens_all[gal]['sigmas_l12_err'] = sigmas_l12_err
            surf_dens_all[gal]['indices'] = indices
            surf_dens_all[gal]['masks'] = masks
            surf_dens_all[gal]['counts'] = counts
            surf_dens_all[gal]['global'] = {
                'log_l12': np.log10(np.average(y)),
                'log_lco': np.log10(np.average(x))
            }
            surf_dens_all[gal]['global_err'] = {
                'log_l12': 0.434 * np.sqrt(np.sum(y_err**2)) / np.sum(y),
                'log_lco': 0.434 * np.sqrt(np.sum(x_err**2)) / np.sum(x)
            }
            surf_dens_all[gal]['n_max'] = n_max

            if do_other_maps == True:
                for k in list(sigma_arrays_other.keys()):
                    surf_dens_all[gal][k] = sigmas_other[k]

    return surf_dens_all


def get_write_combos():
    combos = generate_templates_all()
    with open(fname_combos, 'wb') as p:
        pickle.dump(combos, p)
    combos = pickle.load(open(fname_combos, 'rb'))


def get_write_surf_dens_all():
    import reproject_califa as reproj_c

    def get_sigma_mstar(galname):
        fname = reproj_c.fname_mstar_stacked(galname)
        # opened_fits = fits.open(fname)
        res = pickle.load(open(fname, 'rb'))
        return res['mstar']

    def get_sigma_sfr(galname):
        fname = reproj_c.fname_halpha_stacked(galname)
        # opened_fits = fits.open(fname)
        res = pickle.load(open(fname, 'rb'))
        return res['sigma_sfr']

    def get_sigma_sfr_err(galname):
        fname = reproj_c.fname_halpha_stacked(galname)
        # opened_fits = fits.open(fname)
        res = pickle.load(open(fname, 'rb'))
        return res['sigma_sfr_err']

    # def get_bpt_stacked(galname):
    #     return reproj_c.bpt_stacked(galname)
    #
    other_maps = [get_sigma_mstar, get_sigma_sfr, get_sigma_sfr_err]
    other_maps_names = ['sigma_mstar', 'sigma_sfr', 'sigma_sfr_err']
    other_maps_err = [False, False, True]

    surf_dens_all = compute_surface_densities_combos_12um_and_co(
        other_maps, other_maps_names, other_maps_err)
    with open(fname_surf_dens_all_dict, 'wb') as p:
        pickle.dump(surf_dens_all, p)


def get_global(return_names=False):
    '''
    Get global surface densities and their uncertainties:
    log H2 mass surface density (Msun/pc^2)
    log 12um surface density (Lsun/pc^2)
    '''
    galnames_all = list(surf_dens_all.keys())
    galnames_all_original = galnames_all.copy()
    for galname in galnames_all_original:
        inter = get_inter_califa(galname)
        if (inter == 'M') or (galname in [
                'NGC5406', 'NGC2916', 'UGC09476', 'UGC03973'
        ]):
            print("Removing " + str(galname))
            galnames_all.remove(galname)
    # galnames_all.remove('ARP220')
    # galnames_all.remove('NGC2623')

    n_gal = len(galnames_all)
    x = np.zeros(n_gal)
    x_err = np.zeros(n_gal)
    y = np.zeros(n_gal)
    y_err = np.zeros(n_gal)
    pix_areas = np.zeros(n_gal)
    n_pix = np.zeros(n_gal)
    cosi = np.zeros(n_gal)
    i = 0
    for galname in galnames_all:
        x[i] = surf_dens_all[galname]['global']['log_lco']
        x_err[i] = surf_dens_all[galname]['global_err']['log_lco']
        y[i] = surf_dens_all[galname]['global']['log_l12']
        y_err[i] = surf_dens_all[galname]['global_err']['log_l12']
        pix_areas[i] = surf_dens_all[galname]['pix_area_pc2']
        n_pix[i] = surf_dens_all[galname]['npix_globals']
        cosi[i] = np.cos(np.radians(gal_dict[galname]['incl']))
        # print(galname)
        # print(x[i])
        # print(y[i])
        i += 1

    cosi = cosi[x > -1]
    pix_areas = pix_areas[x > -1]
    n_pix = n_pix[x > -1]
    x_err = x_err[x > -1]
    y_err = y_err[x > -1]
    y = y[x > -1]
    galnames_all = np.array(galnames_all)[x > -1]
    x = x[x > -1]
    # y_err *= 2.5
    yt = y_err / .434 * 10**y
    y_err = np.sqrt((2.5 * yt)**2 + (0.0414 * 10**y)**2)
    y_err = 0.434 * y_err / 10**y
    if return_names:
        return x, y, x_err, y_err, pix_areas, n_pix, cosi, galnames_all
    else:
        return x, y, x_err, y_err, pix_areas, n_pix, cosi


def get_xy_pixel_lum():
    '''
    Get pixel luminosities (log units)
    x = log MH2
    y = log Lsun
    '''
    x = []
    x_err = []
    y = []
    y_err = []

    galnames_all = list(surf_dens_all.keys())
    galnames_all.remove('ARP220')
    galnames_all.remove('NGC2623')

    n = 1
    for galname in galnames_all:
        cosi = np.cos(np.radians(gal_dict[galname]['incl']))
        fac = surf_dens_all[galname]['pix_area_pc2'] / cosi
        x.append(surf_dens_all[galname]['sigmas_co'][str(n)] * fac)
        x_err.append(surf_dens_all[galname]['sigmas_co_err'][str(n)] * fac)
        y.append(surf_dens_all[galname]['sigmas_l12'][str(n)] * fac)
        y_err.append(surf_dens_all[galname]['sigmas_l12_err'][str(n)] * fac)
    x = np.hstack(x)
    y = np.hstack(y)
    x_err = np.hstack(x_err)
    y_err = np.hstack(y_err)
    x_err = x_err[x > -1]
    y_err = y_err[x > -1]
    y = y[x > -1]
    x = x[x > -1]

    x_err = 0.434 * x_err / x
    y_err = 0.434 * y_err / y
    x = np.log10(x)
    y = np.log10(y)
    x_err = x_err[x > -1]
    y_err = y_err[x > -1]
    y = y[x > -1]
    x = x[x > -1]
    return x, y, x_err, y_err


def get_xy_pixel_sigmas():
    '''
    Get pixel surface densities
    x = log Msun/pc^2
    y = log Lsun/pc^2
    '''
    x = []
    x_err = []
    y = []
    y_err = []

    galnames_all = list(surf_dens_all.keys())
    galnames_all.remove('ARP220')
    galnames_all.remove('NGC2623')

    n = 1
    for galname in galnames_all:
        # cosi = np.cos(np.radians(gal_dict[galname]['incl']))
        # fac = surf_dens_all[galname]['pix_area_pc2'] / cosi
        x.append(surf_dens_all[galname]['sigmas_co'][str(n)])
        x_err.append(surf_dens_all[galname]['sigmas_co_err'][str(n)])
        y.append(surf_dens_all[galname]['sigmas_l12'][str(n)])
        y_err.append(surf_dens_all[galname]['sigmas_l12_err'][str(n)])
    x = np.hstack(x)
    y = np.hstack(y)
    x_err = np.hstack(x_err)
    y_err = np.hstack(y_err)
    x_err = x_err[x > -1]
    y_err = y_err[x > -1]
    y = y[x > -1]
    x = x[x > -1]

    x_err = 0.434 * x_err / x
    y_err = 0.434 * y_err / y
    x = np.log10(x)
    y = np.log10(y)
    x_err = x_err[x > -1]
    y_err = y_err[x > -1]
    y = y[x > -1]
    x = x[x > -1]
    return x, y, x_err, y_err


def do_global_fit_sigmas(reverse=False):
    '''
    reverse : fit (log Sigma H2) = slope * (log Sigma 12um) + intercept
    '''
    # scatterplot(x, y, '', '', xerr=x_err, yerr=y_err)
    xg, yg, xg_err, yg_err, pix_areas, n_pix, cosi = get_global()
    out_name = 'global_fit_sigmas'
    if reverse == True:
        out_name = 'global_fit_sigmas_reverse'
        global_fit_result = run_linmix(yg,
                                       xg,
                                       yg_err,
                                       xg_err,
                                       parallelize=True)
    else:
        global_fit_result = run_linmix(xg,
                                       yg,
                                       xg_err,
                                       yg_err,
                                       parallelize=True)
    with open("/Users/ryan/Dropbox/mac/wise_w3_vs_co/%s.pk" % (out_name, ),
              "wb") as p:
        pickle.dump(global_fit_result, p)


def do_global_fit_sigmas_lts(reverse=False,
                             mask_fwd_and_rev=True,
                             do_linmix=True):
    # scatterplot(x, y, '', '', xerr=x_err, yerr=y_err)
    xg, yg, xg_err, yg_err, pix_areas, n_pix, cosi = get_global()
    if mask_fwd_and_rev == True:
        p_fwd = lts_linefit(xg, yg, xg_err, yg_err)
        p_rev = lts_linefit(yg, xg, yg_err, xg_err)
        msk = p_fwd.mask & p_rev.mask
    else:
        msk = np.ones(xg.size).astype(bool)

    if do_linmix == True:
        if reverse == True:
            global_fit_result = run_linmix(yg[msk],
                                           xg[msk],
                                           yg_err[msk],
                                           xg_err[msk],
                                           parallelize=True,
                                           nchains=8)
            return global_fit_result
        else:
            global_fit_result = run_linmix(xg[msk],
                                           yg[msk],
                                           xg_err[msk],
                                           yg_err[msk],
                                           parallelize=True,
                                           nchains=8)
            return global_fit_result
    else:
        if reverse == True:
            p_rev = lts_linefit(yg, xg, yg_err, xg_err)
            return p_rev
        else:
            p_fwd = lts_linefit(xg, yg, xg_err, yg_err)
            return p_fwd


def do_pixel_fit_sigmas():
    x, y, x_err, y_err = get_xy_pixel_sigmas()
    pixel_fit_result = run_linmix(x, y, x_err, y_err, parallelize=True)
    with open("/Users/ryan/Dropbox/mac/wise_w3_vs_co/pixel_fit_sigmas.pk",
              "wb") as p:
        pickle.dump(pixel_fit_result, p)


def do_global_fit_lum():
    xg, yg, xg_err, yg_err, pix_areas, n_pix, cosi = get_global()
    global_fit_result = run_linmix(xg + np.log10(pix_areas * n_pix / cosi),
                                   yg + np.log10(pix_areas * n_pix / cosi),
                                   xg_err, yg_err)

    with open(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/global_fit_luminosities.pk",
            "wb") as p:
        pickle.dump(global_fit_result, p)


def do_pixel_fit_lum():
    x, y, x_err, y_err = get_xy_pixel_lum()
    pixel_fit_result = run_linmix(x, y, x_err, y_err, parallelize=True)
    with open(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/pixel_fit_luminosities.pk",
            "wb") as p:
        pickle.dump(pixel_fit_result, p)


def plot_sigmas_and_fits():
    pixel_fit_result = pickle.load(
        open('/Users/ryan/Dropbox/mac/wise_w3_vs_co/pixel_fit_sigmas.pk',
             'rb'))
    global_fit_result = pickle.load(
        open(
            '/Users/ryan/Dropbox/mac/wise_w3_vs_co/global_fit_sigmas_reverse.pk',
            'rb'))

    x, y, x_err, y_err = get_xy_pixel_sigmas()
    xg, yg, xg_err, yg_err, pix_areas, n_pix, cosi = get_global()

    pl.figure()
    pl.scatter(x,
               y,
               marker='+',
               linestyle='None',
               label='Pixels',
               linewidth=0.5,
               alpha=.5)
    scatterplot(2 + np.log10(4.35 / 1.36),
                -.5,
                '',
                '',
                xerr=np.average(x_err),
                yerr=np.average(y_err),
                label='Typical pixel uncertainty')
    scatterplot(xg,
                yg,
                r"$\log\Sigma_\mathrm{H_2}$ [$M_\odot$ pc$^{-2}$]",
                r"$\log\Sigma_\mathrm{12\>\mu m}$ [$L_\odot$ pc$^{-2}$]",
                xerr=xg_err,
                yerr=yg_err,
                label='Global')
    # pl.plot(np.linspace(-1.5, 3 + np.log10(4.35 / 1.36), 10),
    #         np.linspace(-1.5, 3 + np.log10(4.35 / 1.36), 10) *
    #         pixel_fit_result['slope'] +
    #         (pixel_fit_result['intercept'] -
    #          pixel_fit_result['slope'] * np.log10(4.35 / 1.36)),
    #         'k--',
    #         label='Pixel fit')
    t = np.linspace(-1.5, 3 + np.log10(4.35 / 1.36), 10)
    pl.plot(t,
            t * global_fit_result['slope'] + global_fit_result['intercept'],
            'k-',
            label='Global fit')
    pl.xlim(-1.5, 3 + np.log10(4.35 / 1.36))
    pl.ylim(-1.5, 3 + np.log10(4.35 / 1.36))
    pl.legend(loc='best')


def plot_lums_and_fits(xmin=5, xmax=10):
    pixel_fit_result = pickle.load(
        open('/Users/ryan/Dropbox/mac/wise_w3_vs_co/pixel_fit_luminosities.pk',
             'rb'))
    global_fit_result = pickle.load(
        open(
            '/Users/ryan/Dropbox/mac/wise_w3_vs_co/global_fit_luminosities.pk',
            'rb'))

    x, y, x_err, y_err = get_xy_pixel_lum()
    xg, yg, xg_err, yg_err, pix_areas, n_pix, cosi = get_global()
    xg += np.log10(pix_areas * n_pix / cosi)
    yg += np.log10(pix_areas * n_pix / cosi)

    pl.figure()
    pl.scatter(x,
               y,
               marker='+',
               linestyle='None',
               label='Pixels',
               alpha=0.5,
               linewidth=0.5)
    scatterplot(9,
                6,
                '',
                '',
                xerr=np.average(x_err),
                yerr=np.average(y_err),
                label='Typical pixel uncertainty')
    scatterplot(xg,
                yg,
                r"$\log \> L_\mathrm{CO}$ [K km s$^{-1}$ pc$^2$]",
                r"$\log \> L_\mathrm{12\>\mu m}$ [$L_\odot$]",
                xerr=xg_err,
                yerr=yg_err,
                label='Global')
    xline = np.linspace(xmin, xmax + 1, 10)
    pl.plot(xline,
            xline * pixel_fit_result['slope'] + pixel_fit_result['intercept'],
            'k--',
            label='Pixel fit')
    pl.plot(xline,
            xline * global_fit_result['slope'] +
            global_fit_result['intercept'],
            'k-',
            label='Global fit')
    pl.plot(xline,
            xline / .98 + .14 / .98,
            linewidth=1,
            c='k',
            linestyle=':',
            label='Gao et al. 2019')
    pl.xlim(xmin, xmax)
    pl.ylim(xmin, xmax)
    pl.legend(loc='best')


# Make plots from "surf_dens_all"
def surf_dens_plots_basic():
    '''
    Top row (9 panels): log10(Sigma 12um) vs log10(Sigma CO)
    Bottom row: scatter of each relation about global (all detected pixels) fit
    '''
    # pl.figure(figsize=(8, 5))
    fig, axes = pl.subplots(2, 4, sharex=True, figsize=(8, 5))
    for galname in list(surf_dens_all.keys()):
        n_max = surf_dens_all[galname]['n_max']
        i = 1
        for n in [1, 4, 7, 9]:  #range(1, n_max + 1):
            if n > n_max:
                continue
            pl.subplot(2, 4, i)
            # Note: uncertainties are in this dict too, for when we use them later
            # Also note: sigmas_co are in linear units. "global" are in log
            y = surf_dens_all[galname]['sigmas_co'][str(n)]  # * 3.2
            # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
            x = surf_dens_all[galname]['sigmas_l12'][str(n)]
            # y_err = surf_dens_all[galname]['sigmas_l12_err'][str(n)]

            pl.scatter(np.log10(x),
                       np.log10(y),
                       s=3,
                       alpha=0.5,
                       facecolor='none',
                       edgecolor='k',
                       linewidth=0.5)

            y = surf_dens_all[galname]['global']['log_lco']  #+ np.log10(3.2)
            # x_err = surf_dens_all[galname]['global_err']['log_lco']
            x = surf_dens_all[galname]['global']['log_l12']
            # x_err = surf_dens_all[galname]['global_err']['log_lco']

            pl.scatter(x,
                       y,
                       facecolor='None',
                       edgecolor='r',
                       linewidths=0.5,
                       alpha=0.5)
            i += 1

    for i in range(1, 5):
        pl.subplot(2, 4, i)
        # pl.title(r"$n=%i$" % (i, ))
        pl.title(r"$A=%i$ arcsec$^2$" % (i * 6**2, ))
        t = np.linspace(-2, 6, 10)
        pl.plot(
            t, t * global_fit_result['slope'] + global_fit_result['intercept'],
            'k--')
        # pl.plot(
        #     np.linspace(-2, 6, 10),
        #     np.linspace(-2, 6, 10) * pixel_fit_result['slope'] +
        #     (pixel_fit_result['intercept'] -
        #      pixel_fit_result['slope'] * np.log10(3.2)), 'b--')
        pl.xlim(-1.5, 4)
        pl.ylim(-2, 3)
        if i > 1:
            ax = pl.gca()
            ax.tick_params(axis='both',
                           which='both',
                           bottom=True,
                           labelleft=False)
        # if i == 5:
        #     pl.xlabel(r"$\log\> \Sigma(\mathrm{H_2})$ ($M_\odot$ pc$^{-2}$)",
        #               fontsize=12)
        if i == 1:
            # pl.ylabel(
            #     r"$\log\> \Sigma(12\>\mathrm{\mu m})$ ($L_\odot$ pc$^{-2}$)",
            #     fontsize=12)
            pl.ylabel(r"$\log\> \Sigma(\mathrm{H_2})$ ($M_\odot$ pc$^{-2}$)",
                      fontsize=12)

    # Second row: residuals
    dy_all = dict()

    for galname in list(surf_dens_all.keys()):
        n_max = surf_dens_all[galname]['n_max']
        i = 1
        for n in [1, 4, 7, 9]:  #range(1, n_max + 1):
            if n > n_max:
                continue
            if str(n) not in list(dy_all.keys()):
                dy_all[str(n)] = []
            pl.subplot(2, 4, i + 4)
            # Note: uncertainties are in this dict too, for when we use them later
            # Also note: sigmas_co are in linear units. "global" are in log
            y = surf_dens_all[galname]['sigmas_co'][str(n)]
            # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
            x = surf_dens_all[galname]['sigmas_l12'][str(n)]
            # y_err = 0.434 * surf_dens_all[galname]['sigmas_l12_err'][str(
            #     n)] / y
            dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] +
                                global_fit_result['intercept'])
            # dy = np.log10(y) - (np.log10(x) * pixel_fit_result['slope'] +
            #                     (pixel_fit_result['intercept'] -
            #                      pixel_fit_result['slope'] * np.log10(3.2)))
            dy_all[str(n)] += list(dy)
            pl.scatter(np.log10(x),
                       dy,
                       s=3,
                       alpha=0.5,
                       facecolor='none',
                       edgecolor='k',
                       linewidth=0.5)

            i += 1

    i = 1
    for n in [1, 4, 7, 9]:
        pl.subplot(2, 4, i + 4)
        pl.ylim(-1.5, 4)
        pl.xlim(-1.7, 1.7)
        dy_all_n = np.array(dy_all[str(n)])
        dy_all_n = dy_all_n[np.isnan(dy_all_n) == False]
        pl.plot(np.linspace(-2, 5, 10), np.zeros(10), 'k--', linewidth=0.25)
        pl.text(1.1,
                1.,
                r"$\sigma = %2.2f$ dex" % (np.sqrt(np.average(dy_all_n**2))),
                fontsize=9)
        if i > 1:
            ax = pl.gca()
            ax.tick_params(axis='both',
                           which='both',
                           bottom=True,
                           labelleft=False)

        # if i == 5:
        #     pl.xlabel(r"$\log\> \Sigma(\mathrm{H_2})$ ($M_\odot$ pc$^{-2}$)",
        #               fontsize=12)
        if i == 1:
            # pl.ylabel(r"$\Delta \log\> \Sigma(12\>\mathrm{\mu m})$ (dex)",
            #           fontsize=12)
            pl.ylabel(r"$\Delta \log\> \Sigma(\mathrm{H_2})$ (dex)",
                      fontsize=12)
        if i in [1, 2, 3, 4]:
            pl.subplot(2, 4, i)
            ax = pl.gca()
            ax.tick_params(axis='both',
                           which='both',
                           bottom=True,
                           labelbottom=False)
        i += 1

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    pl.tick_params(axis='both',
                   which='both',
                   labelcolor='none',
                   top=False,
                   bottom=False,
                   left=False,
                   right=False)
    pl.grid(False)
    # pl.xlabel(r"$\log\> \Sigma(\mathrm{H_2})$ ($M_\odot$ pc$^{-2}$)",
    #           fontsize=12)
    pl.xlabel(r"$\log\> \Sigma(12\mu\mathrm{m})$ ($L_\odot$ pc$^{-2}$)",
              fontsize=12)


def surf_dens_plots_colour_global(parameter,
                                  parameter_label,
                                  parameter_range=None,
                                  parameter_colours=None,
                                  do_cbar=True):
    '''
    Top row (9 panels): log10(Sigma 12um) vs log10(Sigma CO), colour coded
    by a global parameter

    Bottom row: scatter of each relation about global (all detected pixels) fit,
    colour coded by a global parameter

    Args:
    parameter (dict) : parameter['NGC5000'] = some number (float)

    parameter_label (str) : label for colour bar

    parameter_range (float, float); optional : min and max value of "parameter"
    to plot. Plots full range if parameter_range == None.

    parameter_colours (dict) : parameter_colours[parameter_value] = 'b' for example

    do_cbar (bool) : if True, plot a colour bar
    '''
    pl.figure(figsize=(15, 5))

    # Figure out the range in this parameter over all galaxies
    if parameter_range is not None:
        min_param = parameter_range[0]
        max_param = parameter_range[1]
    else:
        param_all = parameter[list(surf_dens_all.keys())[0]]
        param_all = param_all[np.isnan(param_all) == False]

        min_param = param_all[0]
        max_param = param_all[0]

        for galname in list(surf_dens_all.keys()):
            param_value_gal = parameter[galname]
            if np.isnan(param_value_gal):
                continue

            if param_value_gal > max_param:
                max_param = param_value_gal
            if param_value_gal < min_param:
                min_param = param_value_gal

    done_cbar = False
    for galname in list(surf_dens_all.keys()):
        param_value_gal = parameter[galname]

        if parameter_range is not None:
            if (param_value_gal < min_param) or (param_value_gal > max_param):
                # Skip this galaxy, it's parameter value
                # is outside the specified range
                continue

        if np.isnan(param_value_gal):
            continue

        n_max = surf_dens_all[galname]['n_max']
        for n in range(1, n_max + 1):
            pl.subplot(2, 9, n)
            # Note: uncertainties are in this dict too, for when we use them later
            # Also note: sigmas_co are in linear units. "global" are in log
            x = surf_dens_all[galname]['sigmas_co'][str(n)]
            # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
            y = surf_dens_all[galname]['sigmas_l12'][str(n)]
            # y_err = surf_dens_all[galname]['sigmas_l12_err'][str(n)]

            if parameter_colours is not None:
                pl.scatter(np.log10(x),
                           np.log10(y),
                           s=3,
                           alpha=0.5,
                           c=parameter_colours[param_value_gal])
            else:
                pl.scatter(np.log10(x),
                           np.log10(y),
                           s=3,
                           alpha=0.5,
                           c=np.ones(x.size) * param_value_gal,
                           vmin=min_param,
                           vmax=max_param)

            if (n == 9) and (done_cbar == False):
                if do_cbar:
                    pl.colorbar(label=parameter_label)
                done_cbar = True

            x = surf_dens_all[galname]['global']['log_lco']
            # x_err = surf_dens_all[galname]['global_err']['log_lco']
            y = surf_dens_all[galname]['global']['log_l12']
            # x_err = surf_dens_all[galname]['global_err']['log_lco']

            pl.scatter(x,
                       y,
                       facecolor='None',
                       edgecolor='k',
                       linewidths=0.5,
                       alpha=0.5)

    # Plot the scatter
    for i in range(1, 10):
        pl.subplot(2, 9, i)
        pl.title(r"$n=%i$" % (i, ))
        pl.plot(
            np.linspace(-2, 6, 10),
            np.linspace(-2, 6, 10) * global_fit_result['slope'] +
            global_fit_result['intercept'], 'k--')
        pl.plot(
            np.linspace(-2, 6, 10),
            np.linspace(-2, 6, 10) * pixel_fit_result['slope'] +
            pixel_fit_result['intercept'], 'b--')
        pl.xlim(-1.5, 4)
        pl.ylim(-2, 3)
        if i > 1:
            ax = pl.gca()
            ax.tick_params(axis='both',
                           which='both',
                           bottom=True,
                           labelleft=False)
        if i == 5:
            pl.xlabel(r"$\log\> \Sigma(\mathrm{CO})$ (K km/s pc$^2$/pc$^2$)",
                      fontsize=8)
        if i == 1:
            pl.ylabel(
                r"$\log\> \Sigma(12\>\mathrm{\mu m})$ ($L_\odot$ pc$^{-2}$)",
                fontsize=8)

    # Second row: residuals
    dy_all = dict()

    for galname in list(surf_dens_all.keys()):
        param_value_gal = parameter[galname]

        if parameter_range is not None:
            if (param_value_gal < min_param) or (param_value_gal > max_param):
                # Skip this galaxy, it's parameter value
                # is outside the specified range
                continue

        if np.isnan(param_value_gal):
            continue

        n_max = surf_dens_all[galname]['n_max']
        for n in range(1, n_max + 1):
            if str(n) not in list(dy_all.keys()):
                dy_all[str(n)] = []
            pl.subplot(2, 9, n + 9)
            # Note: uncertainties are in this dict too, for when we use them later
            # Also note: sigmas_co are in linear units. "global" are in log
            x = surf_dens_all[galname]['sigmas_co'][str(n)]
            # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
            y = surf_dens_all[galname]['sigmas_l12'][str(n)]
            y_err = 0.434 * surf_dens_all[galname]['sigmas_l12_err'][str(
                n)] / y
            # dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] + global_fit_result['intercept'])
            dy = np.log10(y) - (np.log10(x) * pixel_fit_result['slope'] +
                                pixel_fit_result['intercept'])
            dy_all[str(n)] += list(dy)

            if parameter_colours is not None:
                pl.scatter(np.log10(x),
                           dy,
                           s=3,
                           alpha=0.5,
                           c=parameter_colours[param_value_gal])
            else:
                pl.scatter(np.log10(x),
                           dy,
                           s=3,
                           alpha=0.5,
                           c=np.ones(x.size) * param_value_gal,
                           vmin=min_param,
                           vmax=max_param)

            # pl.scatter(np.log10(x), dy, s=3, alpha=0.5, c=np.ones(x.size) * param_value_gal, vmin=min_param, vmax=max_param)
            # x = surf_dens_all[galname]['global']['log_lco']
            # # x_err = surf_dens_all[galname]['global_err']['log_lco']
            # y = surf_dens_all[galname]['global']['log_l12']
            # # x_err = surf_dens_all[galname]['global_err']['log_lco']

            # pl.scatter(x, y, facecolor='None', edgecolor='k', linewidths=0.5, alpha=0.5)

    for i in range(1, 10):
        pl.subplot(2, 9, i + 9)
        pl.xlim(-1.5, 4)
        pl.ylim(-1.7, 1.7)
        dy_all_n = np.array(dy_all[str(i)])
        dy_all_n = dy_all_n[np.isnan(dy_all_n) == False]

        pl.text(1.,
                1.1,
                r"$\sigma = %2.2f$ dex" % (np.sqrt(np.average(dy_all_n**2))),
                fontsize=9)
        if i > 1:
            ax = pl.gca()
            ax.tick_params(axis='both',
                           which='both',
                           bottom=True,
                           labelleft=False)
        if i == 5:
            pl.xlabel(r"$\log\> \Sigma(\mathrm{CO})$ (K km/s pc$^2$/pc$^2$)",
                      fontsize=8)
        if i == 1:
            pl.ylabel(r"$\Delta \log\> \Sigma(12\>\mathrm{\mu m})$ (dex)",
                      fontsize=8)
        if i == 9:
            if do_cbar:
                pl.colorbar(label=parameter_label)


def surf_dens_plots_colour_global_single_n(n,
                                           parameter,
                                           parameter_label,
                                           parameter_range=None,
                                           parameter_colours=None,
                                           do_cbar=True,
                                           hist_ranges=None,
                                           hist_range_labels=None):
    '''
    Top row (1 panel): log10(Sigma 12um) vs log10(Sigma CO), colour coded
    by a global parameter

    Middle row (left): log Sigma(12um) observed - log Sigma(12um) predicted,
    where "predicted" is from fit to all pixels. Points colour-coded
    by a global parameter.

    Middle row (right): histogram of these differences. Plots separate
    histograms for different "hist_ranges," labelled by "hist_range_labels."

    Bottom row: same as middle row but they are differences in
    log Sigma(CO) rather than log Sigma(12um).

    Args:
        n_plot (int) : only plot this value of n (recall that surface
            area is calculated over "n" pixels).

        parameter (dict) : parameter['NGC5000'] = some number (float).

        parameter_label (str) : Label for colour bar.

    Optional args:
        parameter_range [float, float]: min and max value of "parameter"
            to plot. Plots full range if parameter_range == None.

        parameter_colours (dict) : E.g. parameter_colours[parameter_value] = 'b'

        do_cbar (bool) : If True, plot a colour bar on top row.

        hist_ranges (list of floats) : [[min1, max1], [min2, max2], ... ]

        hist_range_labels (list of strings) : Label for each hist range.
    '''
    pl.figure(figsize=(7, 8))

    # Figure out the range in this parameter over all galaxies
    if parameter_range is not None:
        min_param = parameter_range[0]
        max_param = parameter_range[1]
    else:
        param_all = parameter[list(surf_dens_all.keys())[0]]
        param_all = param_all[np.isnan(param_all) == False]

        min_param = param_all[0]
        max_param = param_all[0]

        for galname in list(surf_dens_all.keys()):
            param_value_gal = parameter[galname]
            if np.isnan(param_value_gal):
                continue

            if param_value_gal > max_param:
                max_param = param_value_gal
            if param_value_gal < min_param:
                min_param = param_value_gal

    done_cbar = False
    pl.subplot(3, 2, 1)
    for galname in list(surf_dens_all.keys()):
        param_value_gal = parameter[galname]

        if parameter_range is not None:
            if (param_value_gal < min_param) or (param_value_gal > max_param):
                # Skip this galaxy, it's parameter value
                # is outside the specified range
                continue

        if np.isnan(param_value_gal):
            continue

        n_max = surf_dens_all[galname]['n_max']
        # Note: uncertainties are in this dict too, for when we use them later
        # Also note: sigmas_co are in linear units. "global" are in log
        y = surf_dens_all[galname]['sigmas_co'][str(n)]
        # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
        x = surf_dens_all[galname]['sigmas_l12'][str(n)]
        # y_err = surf_dens_all[galname]['sigmas_l12_err'][str(n)]

        if parameter_colours is not None:
            pl.scatter(np.log10(x),
                       np.log10(y),
                       s=3,
                       alpha=0.5,
                       c=parameter_colours[param_value_gal])
        else:
            pl.scatter(np.log10(x),
                       np.log10(y),
                       s=3,
                       alpha=0.5,
                       c=np.ones(x.size) * param_value_gal,
                       vmin=min_param,
                       vmax=max_param)

        if (do_cbar == True) and (done_cbar == False):
            pl.colorbar().set_label(label=parameter_label, size=8)
            done_cbar = True

        y = surf_dens_all[galname]['global']['log_lco']
        # x_err = surf_dens_all[galname]['global_err']['log_lco']
        x = surf_dens_all[galname]['global']['log_l12']
        # x_err = surf_dens_all[galname]['global_err']['log_lco']

        pl.scatter(x,
                   y,
                   facecolor='None',
                   edgecolor='k',
                   linewidths=0.5,
                   alpha=0.5)

    pl.title(r"$n=%i$" % (n, ))

    # Plot the scatter
    # for i in range(1, 10):
    # pl.subplot(2, 9, i)

    pl.plot(
        np.linspace(-2, 6, 10),
        np.linspace(-2, 6, 10) * global_fit_result['slope'] +
        global_fit_result['intercept'], 'k--')
    # pl.plot(
    #     np.linspace(-2, 6, 10),
    #     np.linspace(-2, 6, 10) * pixel_fit_result['slope'] +
    #     pixel_fit_result['intercept'], 'b--')
    pl.ylim(-1.5, 4)
    pl.xlim(-2, 3)
    # if i > 1:
    # ax = pl.gca()
    # ax.tick_params(axis='both', which='both', bottom=True, labelleft=False)
    # if i == 5:
    pl.ylabel(r"$\log\> \Sigma(\mathrm{H_2})$ ($M_\odot$ pc$^{-2}$)",
              fontsize=8)
    # if i == 1:
    pl.xlabel(r"$\log\> \Sigma(12\>\mathrm{\mu m})$ ($L_\odot$ pc$^{-2}$)",
              fontsize=8)

    # Second row: residuals
    dy_all = dict()
    param_all = dict()

    pl.subplot(3, 2, 3)

    for galname in list(surf_dens_all.keys()):
        param_value_gal = parameter[galname]

        if parameter_range is not None:
            if (param_value_gal < min_param) or (param_value_gal > max_param):
                # Skip this galaxy, it's parameter value
                # is outside the specified range
                continue

        if np.isnan(param_value_gal):
            continue

        n_max = surf_dens_all[galname]['n_max']
        # for n in range(1, n_max+1):
        if str(n) not in list(dy_all.keys()):
            dy_all[str(n)] = []
            param_all[str(n)] = []
        # pl.subplot(2, 9, n+9)
        # Note: uncertainties are in this dict too, for when we use them later
        # Also note: sigmas_co are in linear units. "global" are in log
        y = surf_dens_all[galname]['sigmas_co'][str(n)]
        # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
        x = surf_dens_all[galname]['sigmas_l12'][str(n)]
        # y_err = 0.434 * surf_dens_all[galname]['sigmas_l12_err'][str(n)] / y
        # dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] + global_fit_result['intercept'])
        dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] +
                            global_fit_result['intercept'])
        dy_all[str(n)] += list(dy)
        param_all[str(n)] += list(param_value_gal * np.ones(len(list(dy))))

        if parameter_colours is not None:
            pl.scatter(param_value_gal * np.ones(len(list(dy))),
                       dy,
                       s=3,
                       alpha=0.5,
                       c='b')
        else:
            pl.scatter(param_value_gal * np.ones(len(list(dy))),
                       dy,
                       s=3,
                       alpha=0.5,
                       c='b')

        # pl.scatter(np.log10(x), dy, s=3, alpha=0.5, c=np.ones(x.size) * param_value_gal, vmin=min_param, vmax=max_param)
        # x = surf_dens_all[galname]['global']['log_lco']
        # # x_err = surf_dens_all[galname]['global_err']['log_lco']
        # y = surf_dens_all[galname]['global']['log_l12']
        # # x_err = surf_dens_all[galname]['global_err']['log_lco']

        # pl.scatter(x, y, facecolor='None', edgecolor='k', linewidths=0.5, alpha=0.5)
    pl.plot(np.linspace(min_param, max_param, 10),
            np.zeros(10),
            'k--',
            linewidth=0.5)

    # for i in range(1, 10):
    #     pl.subplot(2, 9, i+9)
    # pl.xlim(-1.5, 4)
    pl.xlim(min_param, max_param)
    ymin_dy = -1.7
    ymax_dy = 1.7
    pl.ylim(ymin_dy, ymax_dy)
    dy_all_n = np.array(dy_all[str(n)])
    param_all_n = np.array(param_all[str(n)])
    param_all_n = param_all_n[np.isnan(dy_all_n) == False]
    dy_all_n = dy_all_n[np.isnan(dy_all_n) == False]

    pl.text((1. + 1.5) / (1.5 + 4.) * (max_param - min_param) + min_param,
            1.1,
            r"$\sigma = %2.2f$ dex" % (np.sqrt(np.average(dy_all_n**2))),
            fontsize=9)
    # if i > 1:
    # ax = pl.gca()
    # ax.tick_params(axis='both', which='both', bottom=True, labelleft=False)
    # if i == 5:
    pl.xlabel(parameter_label, fontsize=8)
    # if i == 1:
    pl.ylabel(r"$\Delta \log\> \Sigma(\mathrm{H_2})$ (dex)", fontsize=8)
    # pl.colorbar(label=parameter_label)

    pl.subplot(3, 2, 4)
    pl.plot(np.linspace(0, 1, 10), np.zeros(10), 'k--', linewidth=0.5)

    weights = np.ones_like(dy_all_n) / float(len(dy_all_n))
    hist_all = pl.hist(dy_all_n,
                       bins=30,
                       orientation='horizontal',
                       histtype='step',
                       color='k',
                       weights=weights,
                       label='All')
    bins_all = hist_all[1]

    if hist_ranges is not None:
        i = 0
        dy_bin = []
        for param_range in hist_ranges:
            dy_param_range = dy_all_n[(param_all_n >= param_range[0])
                                      & (param_all_n <= param_range[1])]
            dy_bin.append(dy_param_range)
            weights = np.ones_like(dy_param_range) / float(len(dy_param_range))
            hist_label = hist_range_labels[i]
            pl.hist(dy_param_range,
                    bins=bins_all,
                    orientation='horizontal',
                    histtype='step',
                    weights=weights,
                    label=hist_label)
            i += 1

        # KS test
        n_hist_bins = len(dy_bin)
        combos_temp = list(itertools.combinations(np.arange(n_hist_bins), 2))
        ks_arr = np.zeros(len(combos_temp))
        # ks_labels =
        j = 0
        for cc in combos_temp:
            data1 = dy_bin[cc[0]]
            data2 = dy_bin[cc[1]]
            ks_arr[j] = scipy.stats.ks_2samp(data1,
                                             data2,
                                             alternative='two-sided',
                                             mode='exact')[0]
            print(str(hist_ranges[cc[0]]) + " " + str(hist_ranges[cc[1]]))
            print("KS: %2.2f" % (ks_arr[j], ))
            print("n1 = %i, n2 = %i" % (data1.size, data2.size))
            print("Critical D(alpha=0.05) = %2.2f" % (1.36 * np.sqrt(
                (data1.size + data2.size) / (data1.size * data2.size))))
            print("Critical D(alpha=0.01) = %2.2f" % (1.63 * np.sqrt(
                (data1.size + data2.size) / (data1.size * data2.size))))
            j += 1

    pl.legend(loc='best', fontsize='xx-small')
    pl.tick_params(axis='both', which='both', bottom=True, labelleft=False)

    pl.ylim(ymin_dy, ymax_dy)
    pl.xlim(0, 0.5)
    pl.xlabel("Fraction", fontsize=8)

    # Third row: residuals
    dy_all = dict()
    param_all = dict()
    pl.subplot(3, 2, 5)

    for galname in list(surf_dens_all.keys()):
        param_value_gal = parameter[galname]
        if parameter_range is not None:
            if (param_value_gal < min_param) or (param_value_gal > max_param):
                # Skip this galaxy, it's parameter value
                # is outside the specified range
                continue

        if np.isnan(param_value_gal):
            continue

        n_max = surf_dens_all[galname]['n_max']
        # for n in range(1, n_max+1):
        if str(n) not in list(dy_all.keys()):
            dy_all[str(n)] = []
            param_all[str(n)] = []
        # pl.subplot(2, 9, n+9)
        # Note: uncertainties are in this dict too, for when we use them later
        # Also note: sigmas_co are in linear units. "global" are in log
        y = surf_dens_all[galname]['sigmas_co'][str(n)]
        # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
        x = surf_dens_all[galname]['sigmas_l12'][str(n)]
        # y_err = 0.434 * surf_dens_all[galname]['sigmas_l12_err'][str(n)] / y
        # dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] + global_fit_result['intercept'])
        m = global_fit_result['slope']
        b = global_fit_result['intercept']
        dy = np.log10(x) - ((np.log10(y) - b) / m)
        dy_all[str(n)] += list(dy)
        param_all[str(n)] += list(param_value_gal * np.ones(len(list(dy))))
        if parameter_colours is not None:
            pl.scatter(param_value_gal * np.ones(len(list(dy))),
                       dy,
                       s=3,
                       alpha=0.5,
                       c='b')
        else:
            pl.scatter(param_value_gal * np.ones(len(list(dy))),
                       dy,
                       s=3,
                       alpha=0.5,
                       c='b')

    pl.plot(np.linspace(min_param, max_param, 10),
            np.zeros(10),
            'k--',
            linewidth=0.5)

    # pl.xlim(-1.5, 4)
    pl.xlim(min_param, max_param)

    ymin_dy = -1.7
    ymax_dy = 1.7
    pl.ylim(ymin_dy, ymax_dy)
    dy_all_n = np.array(dy_all[str(n)])
    param_all_n = np.array(param_all[str(n)])
    param_all_n = param_all_n[np.isnan(dy_all_n) == False]
    # dy_all_n = np.array(dy_all[str(n)])
    dy_all_n = dy_all_n[np.isnan(dy_all_n) == False]

    pl.text((1. + 1.5) / (1.5 + 4.) * (max_param - min_param) + min_param,
            1.1,
            r"$\sigma = %2.2f$ dex" % (np.sqrt(np.average(dy_all_n**2))),
            fontsize=9)
    # if i > 1:
    # ax = pl.gca()
    # ax.tick_params(axis='both', which='both', bottom=True, labelleft=False)
    # if i == 5:
    pl.xlabel(parameter_label, fontsize=8)
    # if i == 1:
    pl.ylabel(r"$\Delta \log\> \Sigma(12\mu\mathrm{m})$ (dex)", fontsize=8)
    # pl.colorbar(label=parameter_label)

    pl.subplot(3, 2, 6)
    pl.plot(np.linspace(0, 1, 10), np.zeros(10), 'k--', linewidth=0.5)
    weights = np.ones_like(dy_all_n) / float(len(dy_all_n))
    hist_all = pl.hist(dy_all_n,
                       bins=30,
                       orientation='horizontal',
                       histtype='step',
                       color='k',
                       weights=weights,
                       label='All')
    bins_all = hist_all[1]

    if hist_ranges is not None:
        i = 0
        for param_range in hist_ranges:
            dy_param_range = dy_all_n[(param_all_n >= param_range[0])
                                      & (param_all_n <= param_range[1])]
            weights = np.ones_like(dy_param_range) / float(len(dy_param_range))
            hist_label = hist_range_labels[i]
            pl.hist(dy_param_range,
                    bins=bins_all,
                    orientation='horizontal',
                    histtype='step',
                    weights=weights,
                    label=hist_label)
            i += 1

    pl.ylim(ymin_dy, ymax_dy)
    pl.xlim(0, 0.5)

    ax = pl.gca()
    ax.tick_params(axis='both', which='both', bottom=True, labelleft=False)
    pl.xlabel(r"Fraction", fontsize=8)
    pl.legend(loc='best', fontsize='xx-small')


def surf_dens_plots_colour_resolved_single_n(n,
                                             parameter,
                                             parameter_label,
                                             parameter_range=None,
                                             parameter_colours=None,
                                             do_cbar=True,
                                             hist_ranges=None,
                                             hist_range_labels=None,
                                             return_param=False):
    '''
    Top row (1 panel): log10(Sigma 12um) vs log10(Sigma CO), colour coded
    by a global parameter

    Middle row (left): log Sigma(12um) observed - log Sigma(12um) predicted,
    where "predicted" is from fit to all pixels. Points colour-coded
    by a global parameter.

    Middle row (right): histogram of these differences. Plots separate
    histograms for different "hist_ranges," labelled by "hist_range_labels."

    Bottom row: same as middle row but they are differences in
    log Sigma(CO) rather than log Sigma(12um).

    Args:
        n_plot (int) : only plot this value of n (recall that surface
            area is calculated over "n" pixels).

        parameter (dict) : parameter['NGC5000'] = some number (float).

        parameter_label (str) : Label for colour bar.

    Optional args:
        parameter_range [float, float]: min and max value of "parameter"
            to plot. Plots full range if parameter_range == None.

        parameter_colours (dict) : E.g. parameter_colours[parameter_value] = 'b'

        do_cbar (bool) : If True, plot a colour bar on top row.

        hist_ranges (list of floats) : [[min1, max1], [min2, max2], ... ]

        hist_range_labels (list of strings) : Label for each hist range.
    '''
    pl.figure(figsize=(7, 8))
    surf_dens_all = pickle.load(open(fname_surf_dens_all_dict, 'rb'))
    # Figure out the range in this parameter over all galaxies
    # if parameter_range is not None:
    min_param = parameter_range[0]
    max_param = parameter_range[1]
    alpha_co = 1.  #4.35

    done_cbar = False
    pl.subplot(3, 2, 1)
    for galname in list(surf_dens_all.keys()):
        if str(n) not in surf_dens_all[galname]['sigma_mstar'].keys():
            continue
        # i_plot_1 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_mstar'][str(n)]))[0]
        # i_plot_2 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_sfr'][str(n)]))[0]
        # i_plot = np.intersect1d(i_plot_1, i_plot_2)

        i_plot = np.arange(
            np.hstack(surf_dens_all[galname]['indices'][str(n)]).size)

        if parameter == 'tdepl':
            tt = surf_dens_all[galname]['sigmas_co'][str(n)]  # [i_plot]
            ss = surf_dens_all[galname]['sigma_sfr'][str(n)]  # [i_plot]
            param_value_gal = (tt * alpha_co * 1e6 / ss) / 1e9
        else:
            param_value_gal = surf_dens_all[galname][parameter][str(n)]

        # if parameter_range is not None:
        #     if (param_value_gal < min_param) or (param_value_gal > max_param):
        #         # Skip this galaxy, it's parameter value
        #         # is outside the specified range
        #         continue

        if i_plot.size == 0:
            continue

        n_max = surf_dens_all[galname]['n_max']
        # Note: uncertainties are in this dict too, for when we use them later
        # Also note: sigmas_co are in linear units. "global" are in log
        y = surf_dens_all[galname]['sigmas_co'][str(n)][i_plot]
        # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
        x = surf_dens_all[galname]['sigmas_l12'][str(n)][i_plot]
        # y_err = surf_dens_all[galname]['sigmas_l12_err'][str(n)]

        pl.scatter(np.log10(x),
                   np.log10(y),
                   s=3,
                   alpha=0.5,
                   c=np.log10(param_value_gal)[i_plot],
                   vmin=min_param,
                   vmax=max_param)

        if (do_cbar == True) and (done_cbar == False):
            pl.colorbar().set_label(label=parameter_label, size=8)
            done_cbar = True

        y = surf_dens_all[galname]['global']['log_lco']
        # x_err = surf_dens_all[galname]['global_err']['log_lco']
        x = surf_dens_all[galname]['global']['log_l12']
        # x_err = surf_dens_all[galname]['global_err']['log_lco']

        pl.scatter(x,
                   y,
                   facecolor='None',
                   edgecolor='k',
                   linewidths=0.5,
                   alpha=0.5)

    # Plot the scatter
    pl.plot(np.linspace(-2, 6, 10),
            np.linspace(-2, 6, 10) * global_fit_result['slope'] +
            global_fit_result['intercept'],
            'k--',
            linewidth=1.)
    # pl.plot(np.linspace(-2, 6, 10),
    #         np.linspace(-2, 6, 10) * pixel_fit_result['slope'] +
    #         pixel_fit_result['intercept'],
    #         'b--',
    #         linewidth=1.)
    pl.ylim(-1.5, 4)
    pl.xlim(-2, 3)
    pl.ylabel(r"$\log\> \Sigma(\mathrm{H_2})$ ($M_\odot$ pc$^{-2}$)",
              fontsize=8)
    pl.xlabel(r"$\log\> \Sigma(12\>\mathrm{\mu m})$ ($L_\odot$ pc$^{-2}$)",
              fontsize=8)

    # Second row: residuals
    dy_all = dict()
    param_all = dict()

    pl.subplot(3, 2, 3)
    # observed log Sigma H2 - predicted log Sigma H2
    for galname in list(surf_dens_all.keys()):
        if str(n) not in surf_dens_all[galname]['sigma_mstar'].keys():
            continue
        # i_plot_1 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_mstar'][str(n)]))[0]
        # i_plot_2 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_sfr'][str(n)]))[0]
        # i_plot = np.intersect1d(i_plot_1, i_plot_2)
        i_plot = np.arange(
            np.hstack(surf_dens_all[galname]['indices'][str(n)]).size)

        if parameter == 'tdepl':
            tt = surf_dens_all[galname]['sigmas_co'][str(n)]  # [i_plot]
            ss = surf_dens_all[galname]['sigma_sfr'][str(n)]  # [i_plot]
            param_value_gal = (tt * alpha_co * 1e6 / ss) / 1e9
        else:
            param_value_gal = surf_dens_all[galname][parameter][str(n)]

        # if parameter_range is not None:
        #     if (param_value_gal < min_param) or (param_value_gal > max_param):
        #         # Skip this galaxy, it's parameter value
        #         # is outside the specified range
        #         continue

        if i_plot.size == 0:
            continue

        n_max = surf_dens_all[galname]['n_max']
        # for n in range(1, n_max+1):
        if str(n) not in list(dy_all.keys()):
            dy_all[str(n)] = []
            param_all[str(n)] = []
        # Note: uncertainties are in this dict too, for when we use them later
        # Also note: sigmas_co are in linear units. "global" are in log
        y = surf_dens_all[galname]['sigmas_co'][str(n)][i_plot]
        # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
        x = surf_dens_all[galname]['sigmas_l12'][str(n)][i_plot]
        # y_err = 0.434 * surf_dens_all[galname]['sigmas_l12_err'][str(n)] / y
        # dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] + global_fit_result['intercept'])
        dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] +
                            global_fit_result['intercept'])
        dy_all[str(n)] += list(dy)
        param_all[str(n)] += list(np.log10(param_value_gal[i_plot]))

        pl.scatter(np.log10(param_value_gal[i_plot]),
                   dy,
                   marker='+',
                   c='b',
                   linewidth=0.5,
                   alpha=0.5,
                   s=18)

    pl.plot(np.linspace(min_param, max_param, 10),
            np.zeros(10),
            'k--',
            linewidth=0.5)

    pl.xlim(min_param, max_param)
    ymin_dy = -1.7
    ymax_dy = 1.7
    pl.ylim(ymin_dy, ymax_dy)
    dy_all_n = np.array(dy_all[str(n)])
    param_all_n = np.array(param_all[str(n)])
    param_all_n = param_all_n[~np.isnan(dy_all_n)]
    dy_all_n = dy_all_n[~np.isnan(dy_all_n)]

    pl.text((1. + 1.5) / (1.5 + 4.) * (max_param - min_param) + min_param,
            1.1,
            r"$\sigma = %2.2f$ dex" % (np.sqrt(np.average(dy_all_n**2))),
            fontsize=9)
    pl.xlabel(parameter_label, fontsize=8)
    pl.ylabel(r"$\Delta \log\> \Sigma(\mathrm{H_2})$ (dex)", fontsize=8)

    pl.subplot(3, 2, 4)
    # Histogram of observed log Sigma H2 - predicted log Sigma H2
    pl.plot(np.linspace(0, 1, 10), np.zeros(10), 'k--', linewidth=0.5)
    weights = np.ones_like(dy_all_n) / float(len(dy_all_n))
    hist_all = pl.hist(dy_all_n,
                       bins=30,
                       orientation='horizontal',
                       histtype='step',
                       color='k',
                       weights=weights,
                       label='All')
    bins_all = hist_all[1]

    if hist_ranges is not None:
        i = 0
        for param_range in hist_ranges:
            dy_param_range = dy_all_n[(param_all_n >= param_range[0])
                                      & (param_all_n <= param_range[1])]
            weights = np.ones_like(dy_param_range) / float(len(dy_param_range))
            hist_label = hist_range_labels[i]
            pl.hist(dy_param_range,
                    bins=bins_all,
                    orientation='horizontal',
                    histtype='step',
                    weights=weights,
                    label=hist_label)
            i += 1
    pl.tick_params(axis='both', which='both', bottom=True, labelleft=False)

    pl.ylim(ymin_dy, ymax_dy)
    pl.xlim(0, 0.5)
    pl.xlabel("Fraction", fontsize=8)
    pl.legend(loc='best', fontsize='xx-small')

    # Third row: residuals in x
    pl.subplot(3, 2, 5)
    # observed log Sigma H2 - predicted log Sigma H2
    dy_all = dict()
    param_all = dict()

    x_all = []
    y_all = []
    x_err_all = []
    y_err_all = []

    for galname in list(surf_dens_all.keys()):
        if str(n) not in surf_dens_all[galname]['sigma_mstar'].keys():
            continue
        # i_plot_1 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_mstar'][str(n)]))[0]
        # i_plot_2 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_sfr'][str(n)]))[0]
        # i_plot = np.intersect1d(i_plot_1, i_plot_2)
        i_plot = np.arange(
            np.hstack(surf_dens_all[galname]['indices'][str(n)]).size)

        if parameter == 'tdepl':
            tt = surf_dens_all[galname]['sigmas_co'][str(n)]  # [i_plot]
            ss = surf_dens_all[galname]['sigma_sfr'][str(n)]  # [i_plot]
            param_value_gal = (tt * alpha_co * 1e6 / ss) / 1e9
        else:
            param_value_gal = surf_dens_all[galname][parameter][str(n)]

        # if parameter_range is not None:
        #     if (param_value_gal < min_param) or (param_value_gal > max_param):
        #         # Skip this galaxy, it's parameter value
        #         # is outside the specified range
        #         continue

        if i_plot.size == 0:
            continue

        n_max = surf_dens_all[galname]['n_max']
        # for n in range(1, n_max+1):
        if str(n) not in list(dy_all.keys()):
            dy_all[str(n)] = []
            param_all[str(n)] = []
        # pl.subplot(2, 9, n+9)
        # Note: uncertainties are in this dict too, for when we use them later
        # Also note: sigmas_co are in linear units. "global" are in log
        y = surf_dens_all[galname]['sigmas_co'][str(n)][i_plot]
        # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
        x = surf_dens_all[galname]['sigmas_l12'][str(n)][i_plot]

        x_all.append(x)
        y_all.append(y)
        y_err_all.append(
            surf_dens_all[galname]['sigmas_co_err'][str(n)][i_plot])
        x_err_all.append(
            surf_dens_all[galname]['sigmas_l12_err'][str(n)][i_plot])

        # y_err = 0.434 * surf_dens_all[galname]['sigmas_l12_err'][str(n)] / y
        # dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] + global_fit_result['intercept'])
        m = global_fit_result['slope']
        b = global_fit_result['intercept']
        dy = np.log10(x) - ((np.log10(y) - b) / m)
        dy_all[str(n)] += list(dy)
        param_all[str(n)] += list(np.log10(param_value_gal[i_plot]))
        pl.scatter(np.log10(param_value_gal[i_plot]),
                   dy,
                   marker='+',
                   c='b',
                   linewidth=0.5,
                   alpha=0.5,
                   s=18)

    pl.plot(np.linspace(min_param, max_param, 10),
            np.zeros(10),
            'k--',
            linewidth=0.5)

    # pl.xlim(-1.5, 4)
    pl.xlim(min_param, max_param)

    ymin_dy = -1.7
    ymax_dy = 1.7
    pl.ylim(ymin_dy, ymax_dy)
    dy_all_n = np.array(dy_all[str(n)])
    param_all_n = np.array(param_all[str(n)])

    xy_all = dict()
    xy_all['x'] = np.hstack(x_all)[~np.isnan(dy_all_n)]
    xy_all['y'] = np.hstack(y_all)[~np.isnan(dy_all_n)]
    xy_all['x_err'] = np.hstack(x_err_all)[~np.isnan(dy_all_n)]
    xy_all['y_err'] = np.hstack(y_err_all)[~np.isnan(dy_all_n)]

    param_all_n = param_all_n[~np.isnan(dy_all_n)]
    # dy_all_n = np.array(dy_all[str(n)])
    dy_all_n = dy_all_n[~np.isnan(dy_all_n)]

    pl.text((1. + 1.5) / (1.5 + 4.) * (max_param - min_param) + min_param,
            1.1,
            r"$\sigma = %2.2f$ dex" % (np.sqrt(np.average(dy_all_n**2))),
            fontsize=9)
    # if i > 1:
    # ax = pl.gca()
    # ax.tick_params(axis='both', which='both', bottom=True, labelleft=False)
    # if i == 5:
    pl.xlabel(parameter_label, fontsize=8)
    # if i == 1:
    pl.ylabel(r"$\Delta \log\> \Sigma(12\mu\mathrm{m})$ (dex)", fontsize=8)
    # pl.colorbar(label=parameter_label)

    pl.subplot(3, 2, 6)
    pl.plot(np.linspace(0, 1, 10), np.zeros(10), 'k--', linewidth=0.5)
    weights = np.ones_like(dy_all_n) / float(len(dy_all_n))
    hist_all = pl.hist(dy_all_n,
                       bins=30,
                       orientation='horizontal',
                       histtype='step',
                       color='k',
                       weights=weights,
                       label='All')
    bins_all = hist_all[1]

    if hist_ranges is not None:
        i = 0
        for param_range in hist_ranges:
            dy_param_range = dy_all_n[(param_all_n >= param_range[0])
                                      & (param_all_n <= param_range[1])]
            weights = np.ones_like(dy_param_range) / float(len(dy_param_range))
            hist_label = hist_range_labels[i]
            pl.hist(dy_param_range,
                    bins=bins_all,
                    orientation='horizontal',
                    histtype='step',
                    weights=weights,
                    label=hist_label)
            i += 1

    pl.ylim(ymin_dy, ymax_dy)
    pl.xlim(0, 0.3)

    ax = pl.gca()
    ax.tick_params(axis='both', which='both', bottom=True, labelleft=False)
    pl.xlabel(r"Fraction", fontsize=8)
    pl.legend(loc='best', fontsize='xx-small')

    if return_param == True:
        return param_all_n, xy_all


def surf_dens_plots_colour_resolved_bpt_single_n(n,
                                                 parameter_label='BPT',
                                                 do_cbar=True,
                                                 hist_ranges=None,
                                                 hist_range_labels=None,
                                                 return_param=False):
    '''
    Top row (1 panel): log10(Sigma 12um) vs log10(Sigma CO), colour coded
    by a global parameter

    Middle row (left): log Sigma(12um) observed - log Sigma(12um) predicted,
    where "predicted" is from fit to all pixels. Points colour-coded
    by a global parameter.

    Middle row (right): histogram of these differences. Plots separate
    histograms for different "hist_ranges," labelled by "hist_range_labels."

    Bottom row: same as middle row but they are differences in
    log Sigma(CO) rather than log Sigma(12um).

    Args:
        n_plot (int) : only plot this value of n (recall that surface
            area is calculated over "n" pixels).

        parameter (dict) : parameter['NGC5000'] = some number (float).

        parameter_label (str) : Label for colour bar.

    Optional args:
        parameter_range [float, float]: min and max value of "parameter"
            to plot. Plots full range if parameter_range == None.

        parameter_colours (dict) : E.g. parameter_colours[parameter_value] = 'b'

        do_cbar (bool) : If True, plot a colour bar on top row.

        hist_ranges (list of floats) : [[min1, max1], [min2, max2], ... ]

        hist_range_labels (list of strings) : Label for each hist range.
    '''
    pl.figure(figsize=(7, 8))
    surf_dens_all = pickle.load(open(fname_surf_dens_all_dict, 'rb'))
    # Figure out the range in this parameter over all galaxies
    # if parameter_range is not None:
    alpha_co = 1.  #4.35

    param_values = dict()
    min_param = -1.5
    max_param = 2.5

    done_cbar = False
    pl.subplot(3, 2, 1)
    for galname in list(surf_dens_all.keys()):
        if len(
                glob.glob(
                    '/Users/ryan/venus/shared_data/califa/DR3-stack/%s/%s_result.pk'
                    % (galname, galname))) == 0:
            print("Nothing exists for this galaxy")
            continue
        # if str(n) not in surf_dens_all[galname]['sigma_mstar'].keys():
        #     continue
        # i_plot_1 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_mstar'][str(n)]))[0]
        # i_plot_2 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_sfr'][str(n)]))[0]
        # i_plot = np.intersect1d(i_plot_1, i_plot_2)
        i_plot = np.arange(
            np.hstack(surf_dens_all[galname]['indices'][str(n)]).size)
        # if i_plot.size == 0:
        #     continue

        param_value_gal = reproj_c.bpt_stacked(galname).flatten()[np.hstack(
            surf_dens_all[galname]['indices'][str(n)]).astype(int)][i_plot]
        param_values[galname] = param_value_gal

        n_max = surf_dens_all[galname]['n_max']
        # Note: uncertainties are in this dict too, for when we use them later
        # Also note: sigmas_co are in linear units. "global" are in log
        y = surf_dens_all[galname]['sigmas_co'][str(n)][i_plot]  #* 3.2
        # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
        x = surf_dens_all[galname]['sigmas_l12'][str(n)][i_plot]
        # y_err = surf_dens_all[galname]['sigmas_l12_err'][str(n)]

        pl.scatter(np.log10(x),
                   np.log10(y),
                   s=3,
                   alpha=0.5,
                   c=param_value_gal,
                   vmin=min_param,
                   vmax=max_param)

        if (do_cbar == True) and (done_cbar == False):
            pl.colorbar().set_label(label=parameter_label, size=8)
            done_cbar = True

        y = surf_dens_all[galname]['global']['log_lco']  #+ np.log10(3.2)
        # x_err = surf_dens_all[galname]['global_err']['log_lco']
        x = surf_dens_all[galname]['global']['log_l12']
        # x_err = surf_dens_all[galname]['global_err']['log_lco']

        pl.scatter(x,
                   y,
                   facecolor='None',
                   edgecolor='k',
                   linewidths=0.5,
                   alpha=0.5)

    # pl.title(r"$n=%i$" % (n, ))

    pl.plot(np.linspace(-2, 6, 10),
            np.linspace(-2, 6, 10) * global_fit_result['slope'] +
            global_fit_result['intercept'],
            'k--',
            linewidth=1.)
    # pl.plot(np.linspace(-2, 6, 10),
    #         np.linspace(-2, 6, 10) * pixel_fit_result['slope'] +
    #         (pixel_fit_result['intercept'] -
    #          pixel_fit_result['slope'] * np.log10(4.35 / 1.36)),
    #         'b--',
    #         linewidth=1.)
    pl.ylim(-1.5, 4)
    pl.xlim(-2, 3)
    # if i > 1:
    # ax = pl.gca()
    # ax.tick_params(axis='both', which='both', bottom=True, labelleft=False)
    # if i == 5:
    pl.ylabel(r"$\log\> \Sigma(\mathrm{H_2})$ ($M_\odot$ pc$^{-2}$)",
              fontsize=8)
    # if i == 1:
    pl.xlabel(r"$\log\> \Sigma(12\>\mathrm{\mu m})$ ($L_\odot$ pc$^{-2}$)",
              fontsize=8)

    # Second row: residuals
    dy_all = dict()
    param_all = dict()

    pl.subplot(3, 2, 3)

    dy_bins = dict()
    dy_bins['-1'] = []
    dy_bins['0'] = []
    dy_bins['1'] = []
    dy_bins['2'] = []

    for galname in list(surf_dens_all.keys()):
        if len(
                glob.glob(
                    '/Users/ryan/venus/shared_data/califa/DR3-stack/%s/%s_result.pk'
                    % (galname, galname))) == 0:
            print("Nothing exists for this galaxy")
            continue
        # if str(n) not in surf_dens_all[galname]['sigma_mstar'].keys():
        #     continue
        # i_plot_1 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_mstar'][str(n)]))[0]
        # i_plot_2 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_sfr'][str(n)]))[0]
        # i_plot = np.intersect1d(i_plot_1, i_plot_2)
        i_plot = np.arange(
            np.hstack(surf_dens_all[galname]['indices'][str(n)]).size)

        param_value_gal = param_values[galname]

        # if parameter_range is not None:
        #     if (param_value_gal < min_param) or (param_value_gal > max_param):
        #         # Skip this galaxy, it's parameter value
        #         # is outside the specified range
        #         continue

        if i_plot.size == 0:
            continue

        n_max = surf_dens_all[galname]['n_max']
        # for n in range(1, n_max+1):
        if str(n) not in list(dy_all.keys()):
            dy_all[str(n)] = []
            param_all[str(n)] = []
        # pl.subplot(2, 9, n+9)
        # Note: uncertainties are in this dict too, for when we use them later
        # Also note: sigmas_co are in linear units. "global" are in log
        y = surf_dens_all[galname]['sigmas_co'][str(n)][i_plot]
        # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
        x = surf_dens_all[galname]['sigmas_l12'][str(n)][i_plot]
        # y_err = 0.434 * surf_dens_all[galname]['sigmas_l12_err'][str(n)] / y
        # dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] + global_fit_result['intercept'])
        dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] +
                            global_fit_result['intercept'])
        dy_all[str(n)] += list(dy)
        param_all[str(n)] += list(param_value_gal)
        for kk in range(0, dy.size):
            if np.isnan(param_value_gal[kk]) == False:
                dy_bins[str(int(param_value_gal[kk]))].append(dy[kk])
        pl.scatter(param_value_gal,
                   dy,
                   marker='+',
                   c='b',
                   linewidth=0.5,
                   alpha=0.5,
                   s=18)

        # pl.scatter(np.log10(x), dy, s=3, alpha=0.5, c=np.ones(x.size) * param_value_gal, vmin=min_param, vmax=max_param)
        # x = surf_dens_all[galname]['global']['log_lco']
        # # x_err = surf_dens_all[galname]['global_err']['log_lco']
        # y = surf_dens_all[galname]['global']['log_l12']
        # # x_err = surf_dens_all[galname]['global_err']['log_lco']

        # pl.scatter(x, y, facecolor='None', edgecolor='k', linewidths=0.5, alpha=0.5)
    pl.plot(np.linspace(min_param, max_param, 10),
            np.zeros(10),
            'k--',
            linewidth=0.5)

    dy_bin_means = np.array([
        np.average(np.array(dy_bins[kk])[~np.isnan(np.array(dy_bins[kk]))])
        for kk in dy_bins.keys()
    ])
    dy_bin_err = np.array([
        np.std(np.array(dy_bins[kk])[~np.isnan(np.array(dy_bins[kk]))]) /
        np.sqrt(np.array(dy_bins[kk])[~np.isnan(np.array(dy_bins[kk]))].size)
        for kk in dy_bins.keys()
    ])
    pl.errorbar(np.array([-1, 0, 1, 2]),
                dy_bin_means,
                yerr=dy_bin_err,
                marker='s',
                color='k',
                markeredgecolor='k',
                markerfacecolor='xkcd:goldenrod',
                linewidth=1.5,
                alpha=1,
                label='Mean',
                capsize=2,
                markersize=5,
                ecolor='k',
                elinewidth=1,
                markeredgewidth=1)

    # for i in range(1, 10):
    #     pl.subplot(2, 9, i+9)
    # pl.xlim(-1.5, 4)
    pl.xlim(min_param, max_param)
    ymin_dy = -1.7
    ymax_dy = 1.7
    pl.ylim(ymin_dy, ymax_dy)
    dy_all_n = np.array(dy_all[str(n)])
    param_all_n = np.array(param_all[str(n)])
    param_all_n = param_all_n[~np.isnan(dy_all_n)]
    dy_all_n = dy_all_n[~np.isnan(dy_all_n)]

    pl.text((1. + 1.5) / (1.5 + 4.) * (max_param - min_param) + min_param,
            1.1,
            r"$\sigma = %2.2f$ dex" % (np.sqrt(np.average(dy_all_n**2))),
            fontsize=9)
    # if i > 1:
    # ax = pl.gca()
    # ax.tick_params(axis='both', which='both', bottom=True, labelleft=False)
    # if i == 5:
    pl.xlabel(parameter_label, fontsize=8)
    # if i == 1:
    pl.ylabel(r"$\Delta \log\> \Sigma(\mathrm{H_2})$ (dex)", fontsize=8)
    # pl.colorbar(label=parameter_label)

    pl.subplot(3, 2, 4)
    pl.plot(np.linspace(0, 1, 10), np.zeros(10), 'k--', linewidth=0.5)

    weights = np.ones_like(dy_all_n) / float(len(dy_all_n))
    hist_all = pl.hist(dy_all_n,
                       bins=30,
                       orientation='horizontal',
                       histtype='step',
                       color='k',
                       weights=weights,
                       label='All')
    bins_all = hist_all[1]

    if hist_ranges is not None:
        i = 0
        for param_range in hist_ranges:
            dy_param_range = dy_all_n[(param_all_n >= param_range[0])
                                      & (param_all_n <= param_range[1])]
            weights = np.ones_like(dy_param_range) / float(len(dy_param_range))
            hist_label = hist_range_labels[i]
            pl.hist(dy_param_range,
                    bins=bins_all,
                    orientation='horizontal',
                    histtype='step',
                    weights=weights,
                    label=hist_label)
            i += 1
    pl.tick_params(axis='both', which='both', bottom=True, labelleft=False)

    pl.ylim(ymin_dy, ymax_dy)
    pl.xlim(0, 0.5)
    pl.xlabel("Fraction", fontsize=8)
    pl.legend(loc='best', fontsize='xx-small')
    # Third row: residuals
    dy_all = dict()
    param_all = dict()
    pl.subplot(3, 2, 5)

    for galname in list(surf_dens_all.keys()):
        if len(
                glob.glob(
                    '/Users/ryan/venus/shared_data/califa/DR3-stack/%s/%s_result.pk'
                    % (galname, galname))) == 0:
            print("Nothing exists for this galaxy")
            continue
        # if str(n) not in surf_dens_all[galname]['sigma_mstar'].keys():
        #     continue
        # i_plot_1 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_mstar'][str(n)]))[0]
        # i_plot_2 = np.where(
        #     ~np.isnan(surf_dens_all[galname]['sigma_sfr'][str(n)]))[0]
        # i_plot = np.intersect1d(i_plot_1, i_plot_2)
        i_plot = np.arange(
            np.hstack(surf_dens_all[galname]['indices'][str(n)]).size)

        param_value_gal = param_values[
            galname]  # surf_dens_all[galname][parameter][str(n)]

        # if parameter_range is not None:
        #     if (param_value_gal < min_param) or (param_value_gal > max_param):
        #         # Skip this galaxy, it's parameter value
        #         # is outside the specified range
        #         continue

        if i_plot.size == 0:
            continue

        n_max = surf_dens_all[galname]['n_max']
        # for n in range(1, n_max+1):
        if str(n) not in list(dy_all.keys()):
            dy_all[str(n)] = []
            param_all[str(n)] = []
        # pl.subplot(2, 9, n+9)
        # Note: uncertainties are in this dict too, for when we use them later
        # Also note: sigmas_co are in linear units. "global" are in log
        x = surf_dens_all[galname]['sigmas_co'][str(n)][i_plot] * 3.2
        # x_err = surf_dens_all[galname]['sigmas_co_err'][str(n)]
        y = surf_dens_all[galname]['sigmas_l12'][str(n)][i_plot]
        # y_err = 0.434 * surf_dens_all[galname]['sigmas_l12_err'][str(n)] / y
        # dy = np.log10(y) - (np.log10(x) * global_fit_result['slope'] + global_fit_result['intercept'])
        m = global_fit_result['slope']
        b = (global_fit_result['intercept'] -
             global_fit_result['slope'] * np.log10(4.35 / 1.36)
             )  # global_fit_result['intercept']
        dy = np.log10(x) - ((np.log10(y) - b) / m)
        dy_all[str(n)] += list(dy)
        param_all[str(n)] += list(param_value_gal)
        pl.scatter(param_value_gal,
                   dy,
                   marker='+',
                   c='b',
                   linewidth=0.5,
                   alpha=0.5,
                   s=18)

    pl.plot(np.linspace(min_param, max_param, 10),
            np.zeros(10),
            'k--',
            linewidth=0.5)

    # pl.xlim(-1.5, 4)
    pl.xlim(min_param, max_param)

    ymin_dy = -1.7
    ymax_dy = 1.7
    pl.ylim(ymin_dy, ymax_dy)
    dy_all_n = np.array(dy_all[str(n)])
    param_all_n = np.array(param_all[str(n)])
    param_all_n = param_all_n[np.isnan(dy_all_n) == False]
    # dy_all_n = np.array(dy_all[str(n)])
    dy_all_n = dy_all_n[np.isnan(dy_all_n) == False]

    pl.text((1. + 1.5) / (1.5 + 4.) * (max_param - min_param) + min_param,
            1.1,
            r"$\sigma = %2.2f$ dex" % (np.sqrt(np.average(dy_all_n**2))),
            fontsize=9)
    # if i > 1:
    # ax = pl.gca()
    # ax.tick_params(axis='both', which='both', bottom=True, labelleft=False)
    # if i == 5:
    pl.xlabel(parameter_label, fontsize=8)
    # if i == 1:
    pl.ylabel(r"$\Delta \log\> \Sigma(\mathrm{CO})$ (dex)", fontsize=8)
    # pl.colorbar(label=parameter_label)

    pl.subplot(3, 2, 6)
    pl.plot(np.linspace(0, 1, 10), np.zeros(10), 'k--', linewidth=0.5)
    weights = np.ones_like(dy_all_n) / float(len(dy_all_n))
    hist_all = pl.hist(dy_all_n,
                       bins=30,
                       orientation='horizontal',
                       histtype='step',
                       color='k',
                       weights=weights,
                       label='All')
    bins_all = hist_all[1]

    if hist_ranges is not None:
        i = 0
        for param_range in hist_ranges:
            dy_param_range = dy_all_n[(param_all_n >= param_range[0])
                                      & (param_all_n <= param_range[1])]
            weights = np.ones_like(dy_param_range) / float(len(dy_param_range))
            hist_label = hist_range_labels[i]
            pl.hist(dy_param_range,
                    bins=bins_all,
                    orientation='horizontal',
                    histtype='step',
                    weights=weights,
                    label=hist_label)
            i += 1

    pl.ylim(ymin_dy, ymax_dy)
    pl.xlim(0, 0.3)

    ax = pl.gca()
    ax.tick_params(axis='both', which='both', bottom=True, labelleft=False)
    pl.xlabel(r"Fraction", fontsize=8)
    pl.legend(loc='best', fontsize='xx-small')

    if return_param == True:
        return param_all_n


def plot_sigma_fit_params_vs_global_logmstar(
        which_fit='linmix_fit_reverse',
        which_alpha='fixed',
        return_data=False,
        gals_to_label=['ARP220', 'NGC2623', 'NGC4676A']):
    '''
    Args:
    which_alpha (str) : either 'fixed', 'fixed_sf_only', or 'met_alpha_sf_only'
    '''

    if return_data == True:
        data = dict()
        data['name'] = []
        data['log_sigma_h2'] = []
        data['log_sigma_h2_err'] = []
        data['log_sigma_12'] = []
        data['log_sigma_12_err'] = []
        data['slope'] = []
        data['slope_err'] = []
        data['intercept'] = []
        data['intercept_err'] = []
        data['i_edge_detable'] = []
        data['dont_fit'] = []

        # Global properties to get
        data['logm_global'] = []
        data['sfr_global'] = []
        data['met_global'] = []
        data['ur_global'] = []
        data['interacting'] = []

    pl.figure(figsize=(8, 4))
    labelled_mergers = False
    labelled_exclude = False
    labelled_gals = False
    for galname in list(surf_dens_all.keys()):
        if len(
                glob.glob(
                    '/Users/ryan/venus/shared_data/califa/DR3-stack/%s/%s_result.pk'
                    % (galname, galname))) == 0:
            print("Nothing exists for this galaxy")
            continue

        gal_dict_i = gal_dict[galname]
        pixel_area_pc2 = surf_dens_all[galname]['pix_area_pc2']

        dont_fit = False
        if galname in ['NGC5406', 'NGC2916', 'UGC09476']:
            dont_fit = True

        if which_fit.split('_')[0] == 'linmix':
            check_type = dict
        if which_fit.split('_')[0] == 'lts':
            check_type = lts_linefit
        if which_fit.split('_')[0] == 'fit':
            check_type = dict

        # Which alpha_co do you want?
        if which_alpha == 'fixed':
            fit_i = gal_dict_i
            x_fit = gal_dict[galname]['x_fit']
            x_err_fit = gal_dict[galname]['x_err_fit']
            y_fit = gal_dict[galname]['y_fit']
            y_err_fit = gal_dict[galname]['y_err_fit']
        if which_alpha == 'fixed_sf_only':
            fit_i = gal_dict_i['fits_fixed_alpha_sf_only']
            x_fit = gal_dict[galname]['x_fit_sf']
            x_err_fit = gal_dict[galname]['x_fit_sf_err']
            y_fit = gal_dict[galname]['y_fit_sf']
            y_err_fit = gal_dict[galname]['y_fit_sf_err']
        if which_alpha == 'met_alpha_sf_only':
            fit_i = gal_dict_i['result_met_alpha_sf_only']
            x_fit = gal_dict[galname]['x_fit_met']
            x_err_fit = gal_dict[galname]['x_fit_met_err']
            y_fit = gal_dict[galname]['y_fit_met']
            y_err_fit = gal_dict[galname]['y_fit_met_err']

        if (type(fit_i[which_fit]) == check_type):
            if check_type == dict:
                slope_best = fit_i[which_fit]['slope']
                intercept_best = fit_i[which_fit]['intercept']
                slope_err = fit_i[which_fit]['slope_err']
                intercept_err = fit_i[which_fit]['intercept_err']
                if which_fit.split('_')[-1] == 'masked':
                    if np.where(fit_i['lts_fit_mask'] == True)[0].size < 6:
                        continue
            else:
                intercept_best, slope_best = fit_i[which_fit].ab
                intercept_err, slope_err = fit_i[which_fit].ab_err
            # intercept_best = intercept_best - slope_best * np.median(gal_dict_i['y_fit'])

            # Get global properties from EDGE table
            i_edge_detable = np.where(edge_detable_names == galname)[0][0]
            logm_global_gal = log_mstar_edge[edge_detable_names == galname]
            logsfr_global_gal = log_sfr_edge[edge_detable_names == galname]
            ur_global_gal = ur_color_edge[edge_detable_names == galname]
            met_global_gal = metallicity_edge[edge_detable_names == galname]

            multiple_gal = get_inter_califa(galname)
            marker = 'o'
            ecolor = 'b'
            if multiple_gal == 'M':
                marker = 's'
                ecolor = 'r'
            if dont_fit == True:
                ecolor = 'g'
                marker = 'x'

            print("%s: slope=%2.2f, intercept=%2.2f, logM*=%2.2f" %
                  (galname, slope_best, intercept_best, logm_global_gal[0]))
            label = None
            if (dont_fit == False) and (labelled_gals == False) and (
                    multiple_gal != 'M'):
                label = 'Indiv. galaxies'
                labelled_gals = True
            if (dont_fit == True) and (labelled_exclude == False) and (
                    multiple_gal != 'M'):
                label = 'Excluded'
                labelled_exclude = True
            if (dont_fit == False) and (multiple_gal == 'M') and (
                    labelled_mergers == False):
                label = 'Pair/merger'
                labelled_mergers = True

            pl.subplot(121)
            scatterplot(logm_global_gal,
                        slope_best,
                        r"$\log \> M_*/M_\odot$",
                        r"Best-fit $N$",
                        yerr=slope_err,
                        marker=marker,
                        ecolor=ecolor,
                        label=label,
                        alpha=0.7)

            if galname in gals_to_label:
                if galname != 'ARP220':
                    pl.text(logm_global_gal + 0.05,
                            slope_best,
                            galname,
                            fontsize=8,
                            weight='bold')
                else:
                    nn = gals_to_label.index(galname)
                    pl.annotate(galname,
                                xy=(logm_global_gal, slope_best),
                                weight='bold',
                                xytext=(logm_global_gal, -0.5 + 0.1 * nn),
                                fontsize=8,
                                arrowprops=dict(arrowstyle="->", color='k'))

            pl.subplot(122)
            scatterplot(logm_global_gal,
                        intercept_best,
                        r"$\log \> M_*/M_\odot$",
                        r"Best-fit $\log \> C$",
                        yerr=intercept_err,
                        marker=marker,
                        ecolor=ecolor,
                        label=label,
                        alpha=0.7)

            if galname in gals_to_label:
                if galname != 'ARP220':
                    pl.text(logm_global_gal + 0.05,
                            intercept_best,
                            galname,
                            fontsize=8,
                            weight='bold')
                else:
                    nn = gals_to_label.index(galname)
                    pl.annotate(galname,
                                xy=(logm_global_gal, intercept_best),
                                weight='bold',
                                xytext=(logm_global_gal, -1 + 0.1 * nn),
                                fontsize=8,
                                arrowprops=dict(arrowstyle="->", color='k'))

            if return_data == True:

                data['name'].append(galname)
                data['dont_fit'].append(dont_fit)
                # data['log_sigma_h2'].append(gal_dict[galname]['x_fit'])
                # data['log_sigma_h2_err'].append(gal_dict[galname]['x_err_fit'])
                # data['log_sigma_12'].append(gal_dict[galname]['y_fit'])
                # data['log_sigma_12_err'].append(gal_dict[galname]['y_err_fit'])
                data['log_sigma_h2'].append(x_fit)
                data['log_sigma_h2_err'].append(x_err_fit)
                data['log_sigma_12'].append(y_fit)
                data['log_sigma_12_err'].append(y_err_fit)
                data['slope'].append(slope_best)
                data['slope_err'].append(slope_err)
                data['intercept'].append(intercept_best)
                data['intercept_err'].append(intercept_err)
                data['i_edge_detable'].append(i_edge_detable)
                data['interacting'].append(multiple_gal[0])
                data['sfr_global'].append(logsfr_global_gal[0])
                data['met_global'].append(met_global_gal[0])
                data['ur_global'].append(float(ur_global_gal[0]))
                data['logm_global'].append(logm_global_gal[0])

    pl.subplot(121)
    # Plot typical logM* uncertainty
    pl.errorbar(9.75,
                -0.75,
                xerr=0.15,
                marker='s',
                color='k',
                markeredgecolor='k',
                markerfacecolor='xkcd:grey',
                linewidth=1.5,
                alpha=1,
                capsize=2,
                markersize=5,
                ecolor='k',
                elinewidth=1,
                markeredgewidth=1)
    pl.subplot(122)
    # Plot typical logM* uncertainty
    pl.errorbar(9.75,
                -1.85,
                xerr=0.15,
                marker='s',
                color='k',
                markeredgecolor='k',
                markerfacecolor='xkcd:grey',
                linewidth=1.5,
                alpha=1,
                capsize=2,
                markersize=5,
                ecolor='k',
                elinewidth=1,
                markeredgewidth=1)

    if return_data == True:
        data['name'] = np.array(data['name'])
        data['dont_fit'] = np.array(data['dont_fit'])
        data['slope'] = np.array(data['slope'])
        data['slope_err'] = np.array(data['slope_err'])
        data['intercept'] = np.array(data['intercept'])
        data['intercept_err'] = np.array(data['intercept_err'])
        data['i_edge_detable'] = np.array(data['i_edge_detable'])
        data['logm_global'] = np.array(data['logm_global'])
        data['sfr_global'] = np.array(data['sfr_global'])
        data['met_global'] = np.array(data['met_global']).astype(float)
        data['ur_global'] = np.array(data['ur_global'])
        data['interacting'] = np.array(data['interacting'])
        return FitAndGlobalPropertiesFullSample(data)

    # pl.subplot(121)
    # # Plot typical logM* uncertainty
    # pl.errorbar(9.75,
    #             -0.75,
    #             xerr=0.15,
    #             marker='s',
    #             color='k',
    #             markeredgecolor='k',
    #             markerfacecolor='xkcd:grey',
    #             linewidth=1.5,
    #             alpha=1,
    #             capsize=2,
    #             markersize=5,
    #             ecolor='k',
    #             elinewidth=1,
    #             markeredgewidth=1)
    # pl.subplot(122)
    # # Plot typical logM* uncertainty
    # pl.errorbar(9.75,
    #             -1.85,
    #             xerr=0.15,
    #             marker='s',
    #             color='k',
    #             markeredgecolor='k',
    #             markerfacecolor='xkcd:grey',
    #             linewidth=1.5,
    #             alpha=1,
    #             capsize=2,
    #             markersize=5,
    #             ecolor='k',
    #             elinewidth=1,
    #             markeredgewidth=1)


def fit_sigma_fit_params_mstar_bins(data, param_bins,
                                    param_name='logm_global'):

    res = dict()
    nbins = len(param_bins)
    res['mean_slope'] = np.zeros(nbins)
    res['median_slope'] = np.zeros(nbins)
    res['std_slope'] = np.zeros(nbins)
    res['mean_intercept'] = np.zeros(nbins)
    res['median_intercept'] = np.zeros(nbins)
    res['std_intercept'] = np.zeros(nbins)

    # res['n_logmstar_fit'] = {'low_mstar': dict(), 'high_mstar': dict()}
    # res['logc_logmstar_fit'] = {'low_mstar': dict(), 'high_mstar': dict()}

    res['fit_pixels_in_bin'] = [dict()] * nbins
    res['fit_xy'] = [[]] * nbins
    param_key = 'mean_' + param_name + '_bin'
    res[param_key] = np.zeros(nbins)

    for i in range(0, nbins):
        print(i)
        param_lo = param_bins[i][0]
        param_hi = param_bins[i][1]
        i_param_bin = (data['dont_fit'] == False) & (
            data[param_name] >= param_lo) & (data[param_name] < param_hi) & (
                data['interacting'] != 'M')

        param_bin = data[param_name][i_param_bin]
        err_param_bin = np.ones(
            param_bin.size) * 0.15  # assume 0.15 dex mstar uncertainty

        slope_bin = data['slope'][i_param_bin]
        slope_err_bin = data['slope_err'][i_param_bin]
        intercept_bin = data['intercept'][i_param_bin]
        intercept_err_bin = data['intercept_err'][i_param_bin]

        res['mean_slope'][i] = np.average(slope_bin)
        res['median_slope'][i] = np.median(slope_bin)
        res['std_slope'][i] = np.std(slope_bin) / np.sqrt(slope_bin.size)

        res['mean_intercept'][i] = np.average(intercept_bin)
        res['median_intercept'][i] = np.median(intercept_bin)
        res['std_intercept'][i] = np.std(intercept_bin) / np.sqrt(
            intercept_bin.size)

        res[param_key][i] = np.average(param_bin)
        # log_sigma_h2_bin = np.hstack(
        #     np.array(data['log_sigma_h2'])[i_logm_bin])
        # log_sigma_h2_err_bin = np.hstack(
        #     np.array(data['log_sigma_h2_err'])[i_logm_bin])
        # log_sigma_12_bin = np.hstack(
        #     np.array(data['log_sigma_12'])[i_logm_bin])
        # log_sigma_12_err_bin = np.hstack(
        #     np.array(data['log_sigma_12_err'])[i_logm_bin])
        #
        # result_linmix_bin = run_linmix(log_sigma_12_bin,
        #                                log_sigma_h2_bin,
        #                                log_sigma_12_err_bin,
        #                                log_sigma_h2_err_bin,
        #                                parallelize=True)
        # res['fit_pixels_in_bin'][i] = result_linmix_bin
        # res['fit_xy'][i] = [log_sigma_12_bin,
        #                                log_sigma_h2_bin,
        #                                log_sigma_12_err_bin,
        #                                log_sigma_h2_err_bin]

    return res


def plot_fit(t,
             a,
             b,
             a_err=0,
             b_err=0,
             xin=None,
             yin=None,
             s=None,
             pivot=0,
             ax=None,
             log=False,
             color='b',
             lw=2,
             alpha=0.5,
             text=True,
             fontsize=9,
             **kwargs):
    """
    alpha is used to shade the uncertainties from a_err and b_err
    **kwargs is passed to plt.plot() for the central line only
    the error band has zorder=-10
    """
    if log:
        if pivot == 0:
            pivot = 1
        y = lambda A, B: 10**A * (t / pivot)**B
    else:
        y = lambda A, B: A + B * (t - pivot)
    if ax is None:
        ax = pl
    # the length may vary depending on whether it's a default color
    # (e.g., 'r' or 'orange') or an rgb(a) color, etc, but as far as
    # I can think of none of these would have length 2.
    if len(color) != 2:
        color = (color, color)
    print('in lnr.plot: color =', color)
    ax.plot(t, y(a, b), ls='-', color=color[0], lw=lw, **kwargs)
    if a_err != 0 or b_err != 0:
        # to make it compatible with either one or two values
        a_err = np.array([a_err]).flatten()
        b_err = np.array([b_err]).flatten()
        if a_err.size == 1:
            a_err = [a_err, a_err]
        if b_err.size == 1:
            b_err = [b_err, b_err]
        err = [
            y(a - a_err[0], b - b_err[0]),
            y(a - a_err[0], b + b_err[1]),
            y(a + a_err[1], b - b_err[0]),
            y(a + a_err[1], b + b_err[1])
        ]
        ylo = np.min(err, axis=0)
        yhi = np.max(err, axis=0)
        ax.fill_between(t,
                        ylo,
                        yhi,
                        color=color[1],
                        alpha=alpha,
                        lw=0,
                        edgecolor='none',
                        zorder=-10)
    if s:
        if log:
            ax.plot(t, (1 + s) * y(a, b), ls='--', color=color[0], lw=lw)
            ax.plot(t, y(a, b) / (1 + s), ls='--', color=color[0], lw=lw)
        else:
            ax.plot(t, y(a, b) + s, ls='--', color=color[0], lw=lw)
            ax.plot(t, y(a, b) - s, ls='--', color=color[0], lw=lw)
    ax = pl.gca()
    print(a, a_err, b, b_err)
    if text:
        string = '$y = a + bx$\n'
        string += '$a=%.2g^{+%.2g}_{-%.2g}$\n' % (a, a_err[0], a_err[1])
        string += '$b=%.2g^{+%.2g}_{-%.2g}$\n' % (b, b_err[0], b_err[1])

        if b > 0:
            xt, yt = 0.05, 0.95
        else:
            xt, yt = 0.05, 0.4
        ax.text(xt,
                yt,
                string,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=fontsize)

        txt = '${\\rm Spearman/Pearson}$\n'
        r, p = stats.spearmanr(xin, yin)
        txt += '$r=%.2g\, p=%.2g$\n' % (r, p)
        r, p = stats.pearsonr(xin, yin)
        txt += '$r=%.2g\, p=%.2g$\n' % (r, p)

        if b > 0:
            xt, yt = 0.95, 0.25
        else:
            xt, yt = 0.95, 0.95
        ax.text(xt,
                yt,
                txt,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=fontsize)

    return


def make_n_and_slope_vs_all_global_params_plots(which_fit='linmix_fit_reverse',
                                                show_global=None):
    data = plot_sigma_fit_params_vs_global_logmstar(which_fit=which_fit,
                                                    return_data=True)
    data = data.data
    pl.close()

    plot_sigma_fit_params_vs_other_property(which_fit=which_fit)
    mstar_bins = [[9.5, 10], [10, 10.5], [10.5, 11], [11, 11.6]]
    sfr_bins = [[-1.5, -0.5], [-.5, 0], [0, .5], [.5, 2]]
    ur_bins = [[.8, 1.5], [1.5, 1.75], [1.75, 2], [2, 2.25], [2.25, 2.7]]
    met_bins = [[8.4, 8.45], [8.45, 8.5], [8.5, 8.55], [8.55, 8.6]]

    data_fits = fit_sigma_fit_params_mstar_bins(data,
                                                mstar_bins,
                                                param_name='logm_global')
    plot_data_fits(data_fits,
                   mstar_bins,
                   param_name='logm_global',
                   i=4,
                   j=2,
                   k=1,
                   legend=False)

    data_fits = fit_sigma_fit_params_mstar_bins(data,
                                                sfr_bins,
                                                param_name='sfr_global')
    plot_data_fits(data_fits,
                   sfr_bins,
                   param_name='sfr_global',
                   i=4,
                   j=2,
                   k=3,
                   legend=False)

    data_fits = fit_sigma_fit_params_mstar_bins(data,
                                                ur_bins,
                                                param_name='ur_global')
    plot_data_fits(data_fits,
                   ur_bins,
                   param_name='ur_global',
                   i=4,
                   j=2,
                   k=5,
                   legend=False)

    data_fits = fit_sigma_fit_params_mstar_bins(data,
                                                met_bins,
                                                param_name='met_global')
    plot_data_fits(data_fits,
                   met_bins,
                   param_name='met_global',
                   i=4,
                   j=2,
                   k=7,
                   legend=False)
    if show_global is not None:
        if type(show_global) == dict:
            m_global = show_global['slope']
            m_global_err = show_global['slope_err']
            b_global = show_global['intercept']
            b_global_err = show_global['intercept_err']
        else:
            b_global, m_global = show_global.ab
            b_global_err, m_global_err = show_global.ab_err

        j = 1
        for bb in [mstar_bins, sfr_bins, ur_bins, met_bins]:
            t = np.linspace(bb[0][0], bb[-1][1], 10)
            pl.subplot(4, 2, j)
            pl.plot(t, np.ones(t.size) * m_global, 'k--')
            pl.plot(t, np.ones(t.size) * (m_global + m_global_err), 'r--')
            pl.plot(t, np.ones(t.size) * (m_global - m_global_err), 'r--')
            pl.xlim(bb[0][0], bb[-1][1])

            pl.subplot(4, 2, j + 1)
            pl.plot(t, np.ones(t.size) * b_global, 'k--')
            pl.plot(t, np.ones(t.size) * (b_global + b_global_err), 'r--')
            pl.plot(t, np.ones(t.size) * (b_global - b_global_err), 'r--')
            pl.xlim(bb[0][0], bb[-1][1])

            j += 2


def make_n_logm_slope_logm_plot(which_fit='linmix_fit_reverse'):
    data = plot_sigma_fit_params_vs_global_logmstar(return_data=True,
                                                    which_fit=which_fit)
    data = data.data
    mstar_bins = [[9.5, 10], [10, 10.5], [10.5, 11], [11, 11.6]]
    data_fits = fit_sigma_fit_params_mstar_bins(data, mstar_bins)
    plot_data_fits(data_fits, mstar_bins)
    pl.tight_layout()
    pl.savefig('/Users/ryan/Dropbox/mac/wise_w3_vs_co/n_logm_slope_logm.pdf')


# Do PCA on slope, and global properties
# same for intercept and global properties
def pca_slope_intercept_global():
    '''
    Not finished with this. PCA on logM* and metallicity seems to pick out
    mass metallicity correlation, puts very low weight on slope
    '''

    import sklearn
    from sklearn.decomposition import PCA

    data = plot_sigma_fit_params_vs_global_logmstar(return_data=True)
    data = data.data
    # , data['met_global']
    good = (data['slope_err'] < .75) & (data['dont_fit'] == False) & (
        data['interacting'] == 'I') & (~np.isnan(data['met_global']))
    n_components = 3
    pca_data = np.vstack(
        [data['slope'], data['logm_global'], data['ur_global']]).T
    pca_data = pca_data[good, :]
    pca_data_av = np.average(pca_data, axis=0)
    pca_data -= pca_data_av
    pca = PCA(n_components=n_components)
    pca.fit(pca_data)
    print(pca.explained_variance_ratio_)
    pc = pca.components_[n_components - 1]

    n_iter = int(1e4)
    perturbations = np.zeros((pca_data.shape[0], n_iter))
    for i in range(0, pca_data.shape[0]):
        perturbations[i, :] = np.random.normal(loc=0.,
                                               scale=data['slope_err'][i],
                                               size=n_iter)

    perturbations_mass = np.zeros((pca_data.shape[0], n_iter))
    for i in range(0, pca_data.shape[0]):
        perturbations_mass[i, :] = np.random.normal(
            loc=0., scale=np.ones(pca_data.shape[0]) * 0.15, size=n_iter)

    n_components = 3

    pc_all = np.zeros((n_iter, n_components))
    pca_data = np.vstack(
        [data['slope'], data['logm_global'], data['met_global']]).T
    pca_data = pca_data[good, :]
    pca_data_av = np.average(pca_data, axis=0)
    for n in range(0, n_iter):
        pca_data = np.vstack(
            [data['slope'], data['logm_global'], data['met_global']]).T
        pca_data = pca_data[good, :]
        pca_data[:, 0] += perturbations[:, n]
        pca_data[:, 1] += perturbations_mass[:, n]
        # pca_data_av = np.average(pca_data, axis=0)
        pca_data -= pca_data_av
        pca = PCA(n_components=n_components)
        pca.fit(pca_data)
        print(pca.explained_variance_ratio_)
        pc = pca.components_[n_components - 1]
        pc_all[n] = pc

    pc = np.average(pc_all, axis=0)
    # +  pca_data[:, 4] * pc[4]
    pl.scatter(-1. * (pca_data[:, 1] * pc[1] + pca_data[:, 2] * pc[2]) / pc[0],
               pca_data[:, 0])

    pl.scatter(pca_data_av[0] - 1. * (pca_data[:, 1] * pc[1]) / pc[0],
               pca_data[:, 0] + pca_data_av[0])
    pl.scatter(-1. * (pca_data[:, 1] * pc[1] + pca_data[:, 2] * pc[2]) / pc[0],
               pca_data[:, 0])

    return 0


def get_fit_and_chi2_all_gals():
    '''
    For a given galaxy, get average N and average
    logC in the appropriate global parameter bins.
    The averages are computed excluding the given
    galaxy.

    Then compute chi2 of this fit for each of the
    following lines:
    1. Fit to global points
    2. N, logC from logM* bin
    3. N, logC from u-r bin
    4. N, logC from 12+log O/H bin
    5. N, logC from logSFR bin
    6. N, logC from the fit to this individual galaxy
    '''
    # "Reverse" means y = log Sigma H2, and x = log Sigma 12um
    global_fit_result = pickle.load(
        open(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/global_fit_sigmas_reverse.pk",
            "rb"))
    m_glob = global_fit_result['slope']
    m_err_glob = global_fit_result['slope_err']
    b_glob = global_fit_result['intercept']
    b_err_glob = global_fit_result['intercept_err']

    # m_glob = 0.626  #0.62588236
    # b_glob = 0.667  #0.97830689

    # plot_fit(np.linspace(-.5,2,10), b_glob, m_glob, a_err=b_err_glob,b_err=m_err_glob)
    data = plot_sigma_fit_params_vs_global_logmstar(return_data=True)
    data = data.data
    pl.close()
    mstar_bins = [[9.5, 10], [10, 10.5], [10.5, 11], [11, 11.6]]
    sfr_bins = [[-1.5, -0.5], [-.5, 0], [0, .5], [.5, 1.6]]
    ur_bins = [[.9, 1.5], [1.5, 1.75], [1.75, 2], [2, 2.25], [2.25, 2.6]]
    met_bins = [[8.4, 8.45], [8.45, 8.5], [8.5, 8.55], [8.55, 8.6]]

    good = (data['dont_fit'] == False) & (data['interacting'] != 'M') & (
        ~np.isnan(data['met_global']))

    galnames_good = data['name'][good]
    chi_all = np.zeros((galnames_good.size, 6))
    i = 0
    for galname in galnames_good:
        res_logm = get_mean_params_without_gal(galname,
                                               data,
                                               mstar_bins,
                                               param_name='logm_global')
        res_sfr = get_mean_params_without_gal(galname,
                                              data,
                                              sfr_bins,
                                              param_name='sfr_global')
        res_met = get_mean_params_without_gal(galname,
                                              data,
                                              met_bins,
                                              param_name='met_global')
        res_ur = get_mean_params_without_gal(galname,
                                             data,
                                             ur_bins,
                                             param_name='ur_global')

        m_i = data['slope'][good][i]
        m_err_i = data['slope_err'][good][i]
        b_i = data['intercept'][good][i]
        b_err_i = data['intercept_err'][good][i]

        yy = np.array(data['log_sigma_h2'])[good][i]
        yy_err = np.array(data['log_sigma_h2_err'])[good][i]
        xx = np.array(data['log_sigma_12'])[good][i]
        xx_err = np.array(data['log_sigma_12_err'])[good][i]

        # Calculate chi2 from each fit
        chi2 = lambda x, y, xerr, yerr, m, b: np.sum(
            (y -
             (m * x + b))**2 / 2. / (yerr**2 + m**2 * xerr**2)) / (y.size - 2)

        chi_all[i][0] = chi2(xx, yy, xx_err, yy_err, m_glob, b_glob)
        chi_all[i][1] = chi2(xx, yy, xx_err, yy_err, m_i, b_i)
        chi_all[i][2] = chi2(xx, yy, xx_err, yy_err, res_logm['mean_slope'],
                             res_logm['mean_intercept'])
        chi_all[i][3] = chi2(xx, yy, xx_err, yy_err, res_sfr['mean_slope'],
                             res_sfr['mean_intercept'])
        chi_all[i][4] = chi2(xx, yy, xx_err, yy_err, res_met['mean_slope'],
                             res_met['mean_intercept'])
        chi_all[i][5] = chi2(xx, yy, xx_err, yy_err, res_ur['mean_slope'],
                             res_ur['mean_intercept'])
        i += 1
    return galnames_good, chi_all


def plot_chi2():
    galnames_good, chi_all = get_fit_and_chi2_all_gals()

    pl.hist(chi_all[:, 0], histtype='step', range=(0, 100), label='global')
    pl.hist(chi_all[:, 1], histtype='step', range=(0, 100), label='indiv.')
    pl.hist(chi_all[:, 2], histtype='step', range=(0, 100), label='logm')
    pl.hist(chi_all[:, 3], histtype='step', range=(0, 100), label='logsfr')
    pl.hist(chi_all[:, 4],
            histtype='step',
            range=(0, 100),
            label='metallicity')
    pl.hist(chi_all[:, 5], histtype='step', range=(0, 100), label='u-r')


def plot_tdepl_ssfr(sigma_sfr,
                    sigma_mstar,
                    tdepl,
                    edge_data=None,
                    my_totals=None):
    # Convert everything to Chabrier (Kroupa ~same as Chabrier)
    mstar_corr = 1.
    sfr_corr = 1
    ssfr = sigma_sfr - (sigma_mstar + 6 - np.log10(mstar_corr))
    tdepl = tdepl + 9
    pl.figure(figsize=(8, 4))
    pl.subplot(121)

    if edge_data is not None:
        pl.scatter(edge_data['log_ssfr'],
                   edge_data['log_tdepl_mol'],
                   facecolor='None',
                   edgecolor='k',
                   linewidths=0.5,
                   alpha=1,
                   label='EDGE totals')
    if my_totals is not None:
        pl.scatter(my_totals['log_ssfr'],
                   my_totals['log_tdepl_mol'] + 9,
                   facecolor='None',
                   edgecolor='r',
                   linewidths=0.5,
                   alpha=1,
                   label='My totals')

    scatterplot(ssfr,
                tdepl,
                r"$\log\>\Sigma_\mathrm{SFR}/\Sigma_*$ (yr$^{-1}$)",
                r"$\log\>\tau_\mathrm{depl.,\> mol}$ (yr)",
                label='Pixels')

    m = (7 - 11.8) / (-7.6 + 14)
    b = 11.8 - (m * -14.)

    pl.plot(np.linspace(-15, -7, 10), np.linspace(-15, -7, 10) * m + b, 'k-')
    # pl.xlim(-14, -7.7)
    # pl.ylim(7, 12)
    pl.xlim(-3.5 - 9, 1.5 - 9)
    pl.ylim(-1.5 + 9, 3 + 9)
    pl.legend(loc='best')

    pl.subplot(122)
    m = (8.3 - 10.5) / (-9 + 13)
    b = 10.5 - (m * -13)
    if edge_data is not None:
        pl.scatter(edge_data['log_ssfr'],
                   edge_data['log_tdepl_mol'],
                   facecolor='None',
                   edgecolor='k',
                   linewidths=0.5,
                   alpha=1,
                   label='EDGE totals')
    if my_totals is not None:
        pl.scatter(my_totals['log_ssfr'],
                   my_totals['log_tdepl_mol'] + 9,
                   facecolor='None',
                   edgecolor='r',
                   linewidths=0.5,
                   alpha=1,
                   label='My totals')
    scatterplot(sigma_sfr + np.log10(7.9 / 5.3) -
                (sigma_mstar + 6 - np.log10(0.61)),
                tdepl,
                r"$\log\>\Sigma_\mathrm{SFR}/\Sigma_*$ (yr$^{-1}$)",
                r"$\log\>\tau_\mathrm{depl.,\> mol}$ (yr)",
                label='Pixels')

    pl.plot(np.linspace(-15, -7, 10), np.linspace(-15, -7, 10) * m + b, 'k-')
    pl.xlim(-13, -9)
    pl.ylim(7.5, 11)
    pl.legend(loc='best')

    pl.tight_layout()


if __name__ == '__main__':
    import sys
    pixel_fit_result = pickle.load(
        open("/Users/ryan/Dropbox/mac/wise_w3_vs_co/pixel_fit_sigmas.pk",
             "rb"))
    # global_fit_result = pickle.load(
    #     open("/Users/ryan/Dropbox/mac/wise_w3_vs_co/global_fit_sigmas.pk", "rb"))
    global_fit_result = pickle.load(
        open(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/global_fit_sigmas_reverse.pk",
            "rb"))

    if sys.argv[1] == 'write_master_dict':
        get_write_surf_dens_all()

    # surf_dens_all = pickle.load(open(fname_surf_dens_all_dict, 'rb'))
    if sys.argv[1] == 'indiv_gal_slope_intercept_vs':
        if sys.argv[2] == 'logm':
            make_n_logm_slope_logm_plot()
        if sys.argv[2] == 'all':
            make_n_and_slope_vs_all_global_params_plots(
                which_fit='linmix_fit_reverse', show_global=global_fit_result)
            pl.savefig(
                '/Users/ryan/Dropbox/mac/wise_w3_vs_co/n_and_logc_vs_global_all.pdf'
            )

    if sys.argv[1] == 'do_fit':
        global_fit_sigmas_lts = do_global_fit_sigmas_lts(reverse=True)

        pl.figure()
        global_fit_sigmas_lts = do_global_fit_sigmas_lts(
            reverse=True, mask_fwd_and_rev=False, do_linmix=False)
        pl.xlabel(r"$\log \> \Sigma(\mathrm{12\mu m})$ [$L_\odot$ pc$^{-2}$]")
        pl.ylabel(r"$\log \> \Sigma(\mathrm{H_2})$ [$M_\odot$ pc$^{-2}$]")
        pl.savefig(
            '/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigmas_global_ltsfit.pdf')

        do_global_fit_sigmas()
        do_global_fit_sigmas(reverse=True)
        global_fit_sigmas_lts = do_global_fit_sigmas_lts(reverse=True)

        do_pixel_fit_sigmas()
        do_global_fit_lum()
        do_pixel_fit_lum()

        plot_sigmas_and_fits()
        plot_lums_and_fits()
        pl.savefig(
            '/Users/ryan/Dropbox/mac/wise_w3_vs_co/lum_12_vs_lum_co_full_sample.pdf'
        )
        pl.savefig(
            '/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_12_vs_sigma_co_full_sample.pdf'
        )

    if sys.argv[1] == 'vary_n':
        surf_dens_plots_basic()
        pl.savefig(
            '/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_12_vs_sigma_h2_vary_n.pdf'
        )

        # surf_dens_plots_basic()
        # pl.tight_layout()
        # pl.savefig("/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_all_v2.pdf")

    if sys.argv[1] == 'resolved_bpt':
        surf_dens_plots_colour_resolved_bpt_single_n(
            1,
            hist_ranges=[[-1.5, -.5], [-.25, .25], [.5, 1.5], [1.75, 2.5]],
            hist_range_labels=['SF', 'Inter.', 'LIER', 'Sy'])
        pl.savefig(
            '/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_12_vs_sigma_co_scatter_bpt.pdf'
        )

    if sys.argv[1] == 'resolved':
        n = 1
        sigma_mstar, xy_all = surf_dens_plots_colour_resolved_single_n(
            n,
            'sigma_mstar',
            r"$\log\> \Sigma(M_*)$ ($M_\odot$pc$^{-2}$)",
            parameter_range=[0, 10 - 6],
            return_param=True)
        pl.tight_layout()
        pl.savefig(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_mstar_resolved.pdf"
        )

        sigma_sfr, xy_all = surf_dens_plots_colour_resolved_single_n(
            n,
            'sigma_sfr',
            r"$\log\> \Sigma(\mathrm{SFR})$ ($M_\odot$ yr$^{-1}$ kpc$^{-2}$)",
            parameter_range=[-4, -.5],
            return_param=True)
        sigma_sfr_err, xy_all = surf_dens_plots_colour_resolved_single_n(
            n,
            'sigma_sfr_err',
            r"$\log\> \Sigma(\mathrm{SFR})$ ($M_\odot$ yr$^{-1}$ kpc$^{-2}$)",
            parameter_range=[-5.5, -2.5],
            return_param=True)
        pl.tight_layout()
        pl.savefig(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_sfr_resolved.pdf"
        )

        tdepl, xy_all = surf_dens_plots_colour_resolved_single_n(
            n,
            'tdepl',
            r"$\log\> \tau_\mathrm{depl.}$ (Gyr)",
            parameter_range=[-1, 2],
            return_param=True)
        pl.tight_layout()
        pl.close('all')
        pl.savefig(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_tdepl_resolved.pdf"
        )

        edge_califa_inter = pd.read_csv(
            '/usr/local/lib/python3.7/site-packages/edge_pydb/dat_glob/external/build/califa_inter/califa_inter.csv',
            comment='#')

        edge_detable = pd.read_csv(
            '/usr/local/lib/python3.7/site-packages/edge_pydb/dat_glob/external/build/DETableFinal.csv'
        )
        edge_detable_names = np.array(edge_detable['Name'])
        # loglco_edge = np.log10(np.array(edge_detable[' coMmol']).astype(float) / 4.36)
        edge_detable_dist = np.array(edge_detable[' caDistMpc'])
        edge_detable_z = np.array(edge_detable[' caZgas'])

        logmmol_edge = np.log10(
            np.array(edge_detable[' coMmol']).astype(float) / 4.36 * 4.35)
        logmmol_err_edge = 0.434 * np.array(
            edge_detable[' coeMmol']).astype(float) / np.array(
                edge_detable[' coMmol']).astype(float)

        log_sfr_edge = np.array(
            edge_detable[' caLHacorr']).astype(float) + np.log10(7.9e-42)
        log_mstar_edge = np.array(edge_detable[' caMstars']).astype(float)
        log_tdepl_mol_edge = logmmol_edge - log_sfr_edge
        log_ssfr_edge = log_sfr_edge - log_mstar_edge

        pl.figure()
        sigma_mol = tdepl + 9 - np.log10(4.3) + np.log10(4.4) + sigma_sfr - 6
        scatterplot(sigma_mol, sigma_sfr,
                    r"$\log\>\Sigma_\mathrm{mol}$ ($M_\odot$ pc$^{-2}$)",
                    r"$\log\>\tau_\mathrm{depl.,\> mol}$ (yr)")
        pl.plot(np.linspace(0, 4, 10),
                np.linspace(0, 4, 10) * 1.08 - 3.49, 'k-')

        pl.xlim(.6, 2.2)
        pl.ylim(-3.1, -1.1)

        pl.figure()
        scatterplot(
            sigma_mstar - np.log10(.61), sigma_sfr + np.log10(7.9 / 5.3),
            r"$\log\>\Sigma_*$ ($M_\odot$pc$^{-2}$)",
            r"$\log\>\Sigma_\mathrm{SFR}$ ($M_\odot$yr$^{-1}$kpc$^{-2}$)")
        b = -4.20
        m = (-1 + 4.20) / 3.2
        pl.plot(np.linspace(.5, 3.5, 10),
                np.linspace(.5, 3.5, 10) * m + b, 'k-')
        pl.ylim(-3.5, -.5)
        pl.xlim(.5, 3.5)

        pl.figure()
        scatterplot(
            tdepl + 9 - 6 + sigma_sfr - np.log10(4.3) + np.log10(4.35),
            sigma_sfr + np.log10(7.9 / 5.3),
            r"$\log\>\Sigma_\mathrm{mol.}$ ($M_\odot$pc$^{-2}$)",
            r"$\log\>\Sigma_\mathrm{SFR}$ ($M_\odot$yr$^{-1}$kpc$^{-2}$)")
        b = -3.22
        m = (-1 + 3.22) / 2.2
        pl.plot(np.linspace(-.5, 2.5, 10),
                np.linspace(-.5, 2.5, 10) * m + b, 'k-')
        pl.ylim(-3.5, -.5)
        pl.xlim(-.5, 2.5)

        # Also plot tdepl vs SSFR
        edge_data = dict()
        edge_data['log_mstar'] = log_mstar_edge
        edge_data['log_tdepl_mol'] = log_tdepl_mol_edge
        edge_data['log_ssfr'] = log_ssfr_edge

        # See mstar_test for how this was made
        my_totals = pickle.load(
            open(
                '/Users/ryan/Dropbox/mac/wise_w3_vs_co/total_tdepl_ssfr_mstar_mine.pk',
                'rb'))

        plot_tdepl_ssfr(sigma_sfr,
                        sigma_mstar,
                        tdepl,
                        edge_data=edge_data,
                        my_totals=my_totals)

    if sys.argv[1] == 'global':

        edge_califa = Table.read(
            '/usr/local/lib/python3.7/site-packages/edge_pydb/dat_glob/external/edge_califa.csv',
            format='ascii.ecsv')
        edge_names = np.array(list(edge_califa['Name']))
        # mstar_edge = np.array(edge_califa['caMass'])
        logsfr_edge = np.array(edge_califa['caSFR'])
        mstar_edge = np.array(edge_califa['caMstars'])
        metallicity_edge = np.array(edge_califa['caOH_O3N2'])
        ur_color_edge = np.array(edge_califa['Su']) - np.array(
            edge_califa['Sr'])

        # court = pd.read_csv('/Users/ryan/Dropbox/mac/wise_w3_vs_co/CALIFA_V1200_dispersions.csv', comment='#', header=None)
        # court_names = np.array(court[0])
        califa_sfr_table = pd.read_csv(
            'sfr_table1_catalan_torrecilla_2015.csv', comment='#', header=None)
        califa_sfr_table_names = np.array(
            [''.join(s.split(' ')) for s in list(califa_sfr_table[1])])
        califa_sfr_table_av = np.array(califa_sfr_table[7])

        names_zomgs = np.array(zomgs['name_01'])
        mstar_zomgs = np.array(zomgs['logmass'])
        sfr_zomgs = np.array(zomgs['logsfr'])

        av_sample = dict()
        av_sample_arr = np.zeros(len(list(surf_dens_all.keys())))
        i = 0
        for galname in list(surf_dens_all.keys()):
            av_i = califa_sfr_table_av[califa_sfr_table_names == galname]
            if av_i.size == 0:
                av_i = np.array([0.])
            av_sample[galname] = av_i[0]
            av_sample_arr[i] = av_i[0]
            i += 1

        surf_dens_plots_colour_global(
            av_sample, r"$A_V$",
            parameter_range=[0, 2.6])  #, parameter_range=[9.5,10])
        pl.tight_layout()

        mstar_sample = dict()
        mstar_sample_arr = np.zeros(len(list(surf_dens_all.keys())))
        i = 0
        for galname in list(surf_dens_all.keys()):
            m_i = mstar_zomgs[names_zomgs == galname]
            if m_i.size == 0:
                m_i = mstar_edge[edge_names == galname]
            mstar_sample[galname] = m_i[
                0]  # mstar_edge[edge_names == galname][0]
            mstar_sample_arr[i] = m_i[
                0]  # mstar_edge[edge_names == galname][0]
            # if galname == 'NGC5784':
            #     mstar_sample[galname] = 11.22
            #     mstar_sample_arr[i] = 11.22
            i += 1

        sfr_sample = dict()
        sfr_sample_arr = np.zeros(len(list(surf_dens_all.keys())))
        i = 0
        for galname in list(surf_dens_all.keys()):
            m_i = sfr_zomgs[names_zomgs == galname]
            if m_i.size == 0:
                m_i = logsfr_edge[edge_names == galname]
            sfr_sample[galname] = m_i[
                0]  # mstar_edge[edge_names == galname][0]
            sfr_sample_arr[i] = m_i[0]  # mstar_edge[edge_names == galname][0]
            i += 1

        ssfr_sample = dict()
        ssfr_sample_arr = np.zeros(len(list(surf_dens_all.keys())))
        i = 0
        for galname in list(surf_dens_all.keys()):
            s_i = sfr_zomgs[names_zomgs == galname]
            if s_i.size == 0:
                s_i = logsfr_edge[edge_names == galname]
                m_i = mstar_edge[edge_names == galname]
                ss_i = s_i - m_i
            else:
                m_i = mstar_zomgs[names_zomgs == galname]
                if m_i.size == 0:
                    s_i = logsfr_edge[edge_names == galname]
                    m_i = mstar_edge[edge_names == galname]
                    ss_i = s_i - m_i
                else:
                    ss_i = s_i - m_i

            ssfr_sample[galname] = ss_i[
                0]  # mstar_edge[edge_names == galname][0]
            ssfr_sample_arr[i] = ss_i[
                0]  # mstar_edge[edge_names == galname][0]
            i += 1

        metallicity_sample = dict()
        metallicity_sample_arr = np.zeros(len(list(surf_dens_all.keys())))
        i = 0
        for galname in list(surf_dens_all.keys()):
            metallicity_sample[galname] = metallicity_edge[edge_names ==
                                                           galname][0]
            metallicity_sample_arr[i] = metallicity_edge[edge_names ==
                                                         galname][0]
            i += 1

        ur_sample = dict()
        ur_sample_arr = np.zeros(len(list(surf_dens_all.keys())))
        i = 0
        for galname in list(surf_dens_all.keys()):
            ur_sample[galname] = ur_color_edge[edge_names == galname][0]
            ur_sample_arr[i] = ur_color_edge[edge_names == galname][0]
            i += 1

        # Plot for all n
        surf_dens_plots_colour_global(
            mstar_sample, r"$\log\>M_*/M_\odot$")  #, parameter_range=[9.5,10])
        pl.tight_layout()
        pl.savefig(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_all_mstar_all.pdf"
        )

        surf_dens_plots_colour_global(ur_sample,
                                      r"$u-r$")  #, parameter_range=[9.5,10])
        pl.tight_layout()
        pl.savefig(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_all_ur_all.pdf"
        )

        surf_dens_plots_colour_global(
            metallicity_sample,
            r"$12+\log\>\mathrm{O/H}$")  #, parameter_range=[9.5,10])
        pl.tight_layout()
        pl.savefig(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_all_metallicity_all.pdf"
        )

        # n=1 only

        surf_dens_plots_colour_global_single_n(1,
                                               av_sample,
                                               r"$A_V$ (mag)",
                                               hist_ranges=[[0.01, 2.6]],
                                               hist_range_labels=[r"blah"])
        pl.tight_layout()

        surf_dens_plots_colour_global_single_n(1,
                                               mstar_sample,
                                               r"$\log\>M_*/M_\odot$",
                                               hist_ranges=[
                                                   [8, 10.1],
                                                   [10.1, 10.5],
                                                   [10.5, 12.],
                                               ],
                                               hist_range_labels=[
                                                   r"$\log\>M_*<10.1$",
                                                   r"$10.1<\log\>M_*<10.5$",
                                                   r"$\log\>M_*>10.5$"
                                               ])
        pl.tight_layout()
        pl.savefig(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_all_mstar_n1_scatter.pdf"
        )

        surf_dens_plots_colour_global_single_n(
            1,
            sfr_sample,
            r"$\log\>\mathrm{SFR}$ ($M_\odot$ yr$^{-1}$)",
            hist_ranges=[[-1.1, 2.05]],
            hist_range_labels=[r"blah"])
        pl.tight_layout()
        pl.savefig(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_all_mstar_n1_scatter.pdf"
        )

        surf_dens_plots_colour_global_single_n(
            1,
            ssfr_sample,
            r"$\log\>\mathrm{SSFR}$ (yr$^{-1}$)",
            hist_ranges=[[-11.5, -8.5]],
            hist_range_labels=[r"blah"])
        pl.tight_layout()

        surf_dens_plots_colour_global_single_n(
            1,
            metallicity_sample,
            r"$12+\log\>\mathrm{O/H}$",
            hist_ranges=[
                [7, 8.45],
                [8.45, 8.55],
                [8.55, 10],
            ],
            hist_range_labels=[r"$Z<8.45$", r"$8.45<Z<8.55$", r"$Z>8.55$"])
        pl.tight_layout()
        pl.savefig(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_all_metallicity_n1_scatter.pdf"
        )

        surf_dens_plots_colour_global_single_n(
            1,
            ur_sample,
            r"$u-r$",
            parameter_range=[1, 3],
            hist_ranges=[[1, 2], [2, 3]],
            hist_range_labels=[r"$1<u-r<2$", r"$2<u-r<3$"])
        pl.tight_layout()
        pl.savefig(
            "/Users/ryan/Dropbox/mac/wise_w3_vs_co/sigma_w3_vs_co_all_ur_colour_n1_scatter.pdf"
        )

# merger = PdfFileMerger()
# for pdf in glob.glob(
#         '/Users/ryan/Dropbox/mac/wise_w3_vs_co/spatial_scale_test/*.pdf'):
#     merger.append(open(pdf, 'rb'))
# with open(
#         '/Users/ryan/Dropbox/mac/wise_w3_vs_co/booklet_spatial_scales_v2.pdf',
#         'wb') as fout:
#     merger.write(fout)
