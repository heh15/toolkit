'''
By Ryan Chown
'''

import surface_densities as surfd
import numpy as np
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
import pandas as pd
import scipy

# data = surfd.plot_sigma_fit_params_vs_global_logmstar(
#     which_fit='lts_fit_reverse', which_alpha='met_alpha_sf_only', return_data=True)

# data = surfd.plot_sigma_fit_params_vs_global_logmstar(
#     which_fit='lts_fit_reverse', which_alpha='fixed', return_data=True)

data = surfd.plot_sigma_fit_params_vs_global_logmstar(
    which_fit='linmix_fit_reverse', which_alpha='fixed', return_data=True)


def get_ssfr(galname):
    names = data.data['name']
    logmstar = data.data['logm_global']
    logsfr = data.data['sfr_global']
    return (logsfr - logmstar)[names == galname][0]


def get_av(galname):
    califa_sfr_table = pd.read_csv(
        '/Users/ryan/Dropbox/mac/califa/catalan15/sfr_table1_catalan_torrecilla_2015.csv',
        comment='#',
        header=None)
    califa_sfr_table_names = np.array(
        [''.join(s.split(' ')) for s in list(califa_sfr_table[1])])
    califa_sfr_table_av = np.array(califa_sfr_table[7])
    av = califa_sfr_table_av[califa_sfr_table_names == galname]
    if av.size == 1:
        av = av[0]
    else:
        av = np.nan
    return av


def get_tir(galname):
    califa_sfr_table = pd.read_csv(
        '/Users/ryan/Dropbox/mac/califa/catalan15/sfr_table1_catalan_torrecilla_2015.csv',
        comment='#',
        header=None)
    califa_sfr_table_names = np.array(
        [''.join(s.split(' ')) for s in list(califa_sfr_table[1])])
    califa_sfr_table_tir = np.array(califa_sfr_table[18])
    califa_sfr_table_tir[califa_sfr_table_tir == ' '] = np.nan
    califa_sfr_table_tir = califa_sfr_table_tir.astype(float)
    tir = np.log10(califa_sfr_table_tir[califa_sfr_table_names == galname])
    if tir.size == 1:
        tir = tir[0]
    else:
        tir = np.nan
    return tir


def get_fuv(galname):
    califa_sfr_table = pd.read_csv(
        '/Users/ryan/Dropbox/mac/califa/catalan15/sfr_table1_catalan_torrecilla_2015.csv',
        comment='#',
        header=None)
    califa_sfr_table_names = np.array(
        [''.join(s.split(' ')) for s in list(califa_sfr_table[1])])
    califa_sfr_table_tir = np.array(califa_sfr_table[13])
    califa_sfr_table_tir[califa_sfr_table_tir == ' '] = np.nan
    califa_sfr_table_tir = califa_sfr_table_tir.astype(float)
    tir = np.log10(califa_sfr_table_tir[califa_sfr_table_names == galname])
    if tir.size == 1:
        tir = tir[0]
    else:
        tir = np.nan
    return tir


def get_nuv(galname):
    califa_sfr_table = pd.read_csv(
        '/Users/ryan/Dropbox/mac/califa/catalan15/sfr_table1_catalan_torrecilla_2015.csv',
        comment='#',
        header=None)
    califa_sfr_table_names = np.array(
        [''.join(s.split(' ')) for s in list(califa_sfr_table[1])])
    califa_sfr_table_tir = np.array(califa_sfr_table[15])
    califa_sfr_table_tir[califa_sfr_table_tir == ' '] = np.nan
    califa_sfr_table_tir = califa_sfr_table_tir.astype(float)
    tir = np.log10(califa_sfr_table_tir[califa_sfr_table_names == galname])
    if tir.size == 1:
        tir = tir[0]
    else:
        tir = np.nan
    return tir


def get_ba(galname):
    califa_sfr_table = pd.read_csv(
        '/Users/ryan/Dropbox/mac/califa/catalan15/sfr_table1_catalan_torrecilla_2015.csv',
        comment='#',
        header=None)
    califa_sfr_table_names = np.array(
        [''.join(s.split(' ')) for s in list(califa_sfr_table[1])])
    califa_sfr_table_tir = np.array(califa_sfr_table[4])
    califa_sfr_table_tir[califa_sfr_table_tir == ' '] = np.nan
    califa_sfr_table_tir = califa_sfr_table_tir.astype(float)
    tir = califa_sfr_table_tir[califa_sfr_table_names == galname]
    if tir.size == 1:
        tir = tir[0]
    else:
        tir = np.nan
    return tir


def get_galaxy_density(galname):
    califa_vel = pd.read_csv(
        '/Users/ryan/Dropbox/mac/califa/catalan15/catalan_torrecilla_2017_table1.csv',
        skiprows=53,
        header=None)
    califa_vel_names = np.array(
        [''.join(s.split(' ')).split('*')[0] for s in list(califa_vel[1])])
    veldisp = np.array(califa_vel[15])
    veldisp[veldisp == '      '] = np.nan
    veldisp[veldisp == '       '] = np.nan
    veldisp[veldisp == '           '] = np.nan
    veldisp = veldisp.astype(float)
    veldisp_gal = veldisp[califa_vel_names == galname]
    if veldisp_gal.size == 1:
        veldisp_gal = np.log10(veldisp_gal[0])
    else:
        veldisp_gal = np.nan
    return veldisp_gal


def get_distance_mpc(galname):
    return surfd.gal_dict[galname]['dist_Mpc'][0]


def get_w4(galname):
    califa_sfr_table = pd.read_csv(
        '/Users/ryan/Dropbox/mac/califa/catalan15/sfr_table1_catalan_torrecilla_2015.csv',
        comment='#',
        header=None)
    califa_sfr_table_names = np.array(
        [''.join(s.split(' ')) for s in list(califa_sfr_table[1])])
    califa_sfr_table_tir = np.array(califa_sfr_table[20])
    califa_sfr_table_tir[califa_sfr_table_tir == ' '] = np.nan
    califa_sfr_table_tir = califa_sfr_table_tir.astype(float)
    tir = np.log10(califa_sfr_table_tir[califa_sfr_table_names == galname])
    if tir.size == 1:
        tir = tir[0]
    else:
        tir = np.nan
    return tir


def get_veldisp(galname):
    califa_vel = pd.read_csv(
        '/Users/ryan/Dropbox/mac/califa/catalan15/catalan_torrecilla_2017_table1.csv',
        skiprows=53,
        header=None)
    califa_vel_names = np.array(
        [''.join(s.split(' ')).split('*')[0] for s in list(califa_vel[1])])
    veldisp = np.array(califa_vel[18])
    veldisp[veldisp == '        '] = np.nan
    veldisp[veldisp == '  '] = np.nan
    veldisp[veldisp == '       '] = np.nan
    veldisp = veldisp.astype(float)
    veldisp_gal = veldisp[califa_vel_names == galname]
    if veldisp_gal.size == 1:
        veldisp_gal = veldisp_gal[0]
    else:
        veldisp_gal = np.nan
    return veldisp_gal


def get_stellar_veldisp_queens(galname):
    # https://www.physics.queensu.ca/Astro/people/Stephane_Courteau/gilhuly2017/index.html
    # Stellar velocity dispersion in 5" aperture
    # Other option is 30" aperture
    queens = pd.read_csv(
        '/Users/ryan/Dropbox/mac/wise_w3_vs_co/CALIFA_V1200_dispersions.csv',
        skiprows=14,
        header=None)
    queens_names = np.array(queens[0])
    veldisp = np.array(queens[1])
    veldisp_gal = veldisp[queens_names == galname]
    if veldisp_gal.size == 1:
        veldisp_gal = veldisp_gal[0]
    else:
        veldisp_gal = np.nan
    if veldisp_gal == 0:
        return np.nan
    return veldisp_gal


def get_ha_veldisp_queens(galname):
    # https://www.physics.queensu.ca/Astro/people/Stephane_Courteau/gilhuly2017/index.html
    # Stellar velocity dispersion in 5" aperture
    # Other option is 30" aperture
    queens = pd.read_csv(
        '/Users/ryan/Dropbox/mac/wise_w3_vs_co/CALIFA_V1200_dispersions.csv',
        skiprows=14,
        header=None)
    queens_names = np.array(queens[0])
    veldisp = np.array(queens[5])
    veldisp_gal = veldisp[queens_names == galname]
    if veldisp_gal.size == 1:
        veldisp_gal = veldisp_gal[0]
    else:
        veldisp_gal = np.nan
    if veldisp_gal == 0:
        return np.nan
    return veldisp_gal


def get_re(galname):
    queens = pd.read_csv(
        '/Users/ryan/Dropbox/mac/wise_w3_vs_co/GC2018_CALIFA_photometry_models_corrected.txt',
        delim_whitespace=True,
        comment='#',
        header=None)
    queens_names = np.array(queens[0])
    queens_re = np.array(queens[16])
    if galname in queens_names:
        r_e = queens_re[queens_names == galname][0]
        if r_e == -99.:
            return np.nan
        return r_e
    return np.nan


def get_re_arcsec(galname):
    queens = pd.read_csv(
        '/Users/ryan/Dropbox/mac/wise_w3_vs_co/GC2018_CALIFA_photometry_models_uncorrected.txt',
        delim_whitespace=True,
        comment='#',
        header=None)
    queens_names = np.array(queens[0])
    queens_re = np.array(queens[12])
    if galname in queens_names:
        r_e = queens_re[queens_names == galname][0]
        if r_e == -99.:
            return np.nan
        return r_e
    return np.nan


def get_sersic(galname):
    column_num = 'n_g'  # 'bab_g', 'BTg'
    catalan = fits.open(
        '/Users/ryan/Dropbox/mac/califa/catalan15/photometric_decomposition.fits'
    )
    f_ms = fits.open(
        '/Users/ryan/Dropbox/mac/tsinghua/CALIFA_2_MS_class.fits.txt')
    califa_ms_names = f_ms[1].data['REALNAME']
    califa_ms_ids = f_ms[1].data['CALIFAID']
    if galname in califa_ms_names:
        califa_id = califa_ms_ids[califa_ms_names == galname]
        if califa_id in catalan[1].data['CALIFAID']:
            sersic = catalan[1].data[column_num][catalan[1].data['CALIFAID'] ==
                                                 califa_id][0]
            if sersic < -100:
                sersic = np.nan
            return sersic
        else:
            return np.nan
    else:
        f_es = fits.open('/Users/ryan/venus/home/CALIFA_2_ES_class.fits')
        califa_es_names = f_es[1].data['realname']
        califa_es_ids = f_es[1].data['CALIFAID']
        califa_id = califa_ms_ids[califa_ms_names == galname]
        if califa_id in catalan[1].data['CALIFAID']:
            sersic = catalan[1].data[column_num][catalan[1].data['CALIFAID'] ==
                                                 califa_id][0]
            if sersic < -100:
                sersic = np.nan
            return sersic
        else:
            return np.nan


def get_axis_ratio_bulge(galname):
    column_num = 'bab_g'  #, 'BTg'
    catalan = fits.open(
        '/Users/ryan/Dropbox/mac/califa/catalan15/photometric_decomposition.fits'
    )
    f_ms = fits.open(
        '/Users/ryan/Dropbox/mac/tsinghua/CALIFA_2_MS_class.fits.txt')
    califa_ms_names = f_ms[1].data['REALNAME']
    califa_ms_ids = f_ms[1].data['CALIFAID']
    if galname in califa_ms_names:
        califa_id = califa_ms_ids[califa_ms_names == galname]
        if califa_id in catalan[1].data['CALIFAID']:
            ba = catalan[1].data[column_num][catalan[1].data['CALIFAID'] ==
                                             califa_id][0]
            if ba < -100:
                ba = np.nan
            return ba
        else:
            return np.nan
    else:
        f_es = fits.open('/Users/ryan/venus/home/CALIFA_2_ES_class.fits')
        califa_es_names = f_es[1].data['realname']
        califa_es_ids = f_es[1].data['CALIFAID']
        califa_id = califa_ms_ids[califa_ms_names == galname]
        if califa_id in catalan[1].data['CALIFAID']:
            ba = catalan[1].data[column_num][catalan[1].data['CALIFAID'] ==
                                             califa_id][0]
            if ba < -100:
                ba = np.nan
            return ba
        else:
            return np.nan


def get_incl(galname):
    sfr_mass_cat = pd.read_csv(
        '/Users/ryan/Dropbox/mac/califa/catalan15/sfrVSmass_califa.csv',
        header=None,
        comment='#')
    sfr_mass_cat_names = np.array(sfr_mass_cat[0])
    if galname in sfr_mass_cat_names:
        return np.array(sfr_mass_cat[4])[sfr_mass_cat_names == galname]
    else:
        return np.nan


def get_ionclass(galname):
    sfr_mass_cat = pd.read_csv(
        '/Users/ryan/Dropbox/mac/califa/catalan15/sfrVSmass_califa.csv',
        header=None,
        comment='#')
    sfr_mass_cat_names = np.array(sfr_mass_cat[0])
    if galname in sfr_mass_cat_names:
        return np.array(sfr_mass_cat[3])[sfr_mass_cat_names == galname]
    else:
        return np.nan


def get_axis_ratio_disk(galname):
    column_num = 'bad_g'  #, 'BTg'
    catalan = fits.open(
        '/Users/ryan/Dropbox/mac/califa/catalan15/photometric_decomposition.fits'
    )
    f_ms = fits.open(
        '/Users/ryan/Dropbox/mac/tsinghua/CALIFA_2_MS_class.fits.txt')
    califa_ms_names = f_ms[1].data['REALNAME']
    califa_ms_ids = f_ms[1].data['CALIFAID']
    if galname in califa_ms_names:
        califa_id = califa_ms_ids[califa_ms_names == galname]
        if califa_id in catalan[1].data['CALIFAID']:
            ba = catalan[1].data[column_num][catalan[1].data['CALIFAID'] ==
                                             califa_id][0]
            if ba < -100:
                ba = np.nan
            return ba
        else:
            return np.nan
    else:
        f_es = fits.open('/Users/ryan/venus/home/CALIFA_2_ES_class.fits')
        califa_es_names = f_es[1].data['realname']
        califa_es_ids = f_es[1].data['CALIFAID']
        califa_id = califa_ms_ids[califa_ms_names == galname]
        if califa_id in catalan[1].data['CALIFAID']:
            ba = catalan[1].data[column_num][catalan[1].data['CALIFAID'] ==
                                             califa_id][0]
            if ba < -100:
                ba = np.nan
            return ba
        else:
            return np.nan


def get_bulge_to_total(galname):
    column_num = 'BT_G'
    catalan = fits.open(
        '/Users/ryan/Dropbox/mac/califa/catalan15/photometric_decomposition.fits'
    )
    f_ms = fits.open(
        '/Users/ryan/Dropbox/mac/tsinghua/CALIFA_2_MS_class.fits.txt')
    califa_ms_names = f_ms[1].data['REALNAME']
    califa_ms_ids = f_ms[1].data['CALIFAID']
    if galname in califa_ms_names:
        califa_id = califa_ms_ids[califa_ms_names == galname]
        if califa_id in catalan[1].data['CALIFAID']:
            bt = catalan[1].data[column_num][catalan[1].data['CALIFAID'] ==
                                             califa_id][0]
            if bt < -100:
                bt = np.nan
            return bt
        else:
            return np.nan
    else:
        f_es = fits.open('/Users/ryan/venus/home/CALIFA_2_ES_class.fits')
        califa_es_names = f_es[1].data['realname']
        califa_es_ids = f_es[1].data['CALIFAID']
        califa_id = califa_ms_ids[califa_ms_names == galname]
        if califa_id in catalan[1].data['CALIFAID']:
            bt = catalan[1].data[column_num][catalan[1].data['CALIFAID'] ==
                                             califa_id][0]
            if bt < -100:
                bt = np.nan
            return bt
        else:
            return np.nan


# get_sersic
# get_axis_ratio
# get_bulge_to_total
# 13, 15 20
#
# import linmix
# def run_linmix(x, y, xerr, yerr, nondet=None, parallelize=False):
#     lm_result = linmix.LinMix(x, y, xerr, yerr, K=2, delta=nondet, parallelize=parallelize)
#     lm_result.run_mcmc(silent=True)
#     chains = np.vstack([lm_result.chain['alpha'], lm_result.chain['beta']])
#     # print(np.average(chains[0]), np.average(chains[1]))
#     result = dict()
#     result['chains'] = chains
#     result['intercept'] = np.average(chains[0])
#     result['intercept_err'] = np.std(chains[0])
#     result['slope'] = np.average(chains[1])
#     result['slope_err'] = np.std(chains[1])
#     return result
#
#
# galname = 'NGC6301'
# xe = surfd.gal_dict[galname]['x_err_fit']
# x = surfd.gal_dict[galname]['x_fit']
# y = surfd.gal_dict[galname]['y_fit']
# ye = surfd.gal_dict[galname]['y_err_fit']
#
# fit_result_reverse = run_linmix(y,
#                                 x,
#                                 ye,
#                                 xe,
#                                 parallelize=False)
#
# nondet = xe < 0.434/3
# fit_result_reverse_nondet = run_linmix(y[nondet],
#                                 x[nondet],
#                                 ye[nondet],
#                                 xe[nondet],
#                                 parallelize=False)
#

#
# califa_sfr_table = pd.read_csv(
#     'sfr_table1_catalan_torrecilla_2015.csv', comment='#', header=None)

# lin_fit = fits.open('/Users/ryan/Dropbox/mac/tsinghua/CALIFA_galfit_ryan.fits')
av_bins = [[0.01, 0.5], [0.5, 1.], [1., 1.5], [1.5, 2.5]]
tir_bins = [[-0.25, 0.5], [0.5, 1.], [1., 2.]]
fuv_bins = [[-1., 0.], [0., 0.5], [0.5, 1.], [1., 2]]
nuv_bins = [[-1., 0.], [0., 0.5], [0.5, 1.], [1., 2]]
w4_bins = [[-1, 0.], [0., 1], [1, 2.5]]
data.add_property('av',
                  get_av,
                  property_label=r"$A(\mathrm{H\alpha})$",
                  property_bins=av_bins)
data.add_property('l_tir',
                  get_tir,
                  property_label=r"$\log L_\mathrm{TIR}$",
                  property_bins=tir_bins)
data.add_property('l_fuv',
                  get_fuv,
                  property_label=r"$\log L_\mathrm{FUV}$",
                  property_bins=fuv_bins)
data.add_property('l_nuv',
                  get_nuv,
                  property_label=r"$\log L_\mathrm{NUV}$",
                  property_bins=nuv_bins)
data.add_property('l_w4',
                  get_w4,
                  property_label=r"$\log L_\mathrm{W4}$",
                  property_bins=w4_bins)
data.add_property('ba', get_ba, property_label=r"$b/a$", property_bins=None)

data.add_property('log_ssfr',
                  get_ssfr,
                  property_label=r"$\log \mathrm{SSFR}$",
                  property_bins=None)
data.add_property('n_g',
                  get_sersic,
                  property_label=r"$n_g$",
                  property_bins=None)
data.add_property('ba_g_bulge',
                  get_axis_ratio_bulge,
                  property_label=r"Bulge $(b/a)_g$",
                  property_bins=None)
data.add_property('ba_g_disk',
                  get_axis_ratio_disk,
                  property_label=r"Disk $(b/a)_g$",
                  property_bins=None)
data.add_property('bt_g',
                  get_bulge_to_total,
                  property_label=r"$(B/T)_g$",
                  property_bins=None)
data.add_property('v_disp_bulge',
                  get_veldisp,
                  property_label=r"$\sigma_\mathrm{bulge}$ [km/s]",
                  property_bins=None)
data.add_property('v_disp_stellar',
                  get_stellar_veldisp_queens,
                  property_label=r"Queens $\sigma_\mathrm{*}$ [km/s]",
                  property_bins=None)
data.add_property('v_disp_ha',
                  get_ha_veldisp_queens,
                  property_label=r"Queens $\sigma_\mathrm{H\alpha}$ [km/s]",
                  property_bins=None)
data.add_property('dmpc',
                  get_distance_mpc,
                  property_label=r"$D_L$",
                  property_bins=None)
data.add_property('r_e',
                  get_re_arcsec,
                  property_label=r"$r_e$ [arcsec]",
                  property_bins=None)


def get_n_sf_pix(galname):
    sigma_h2_map = data.map_sigma_h2(galname,
                                     which_alpha='fixed',
                                     plot_map=False,
                                     return_map=True)
    bpt = data.map_bpt(galname, plot_map=False, return_map=True)
    good = (bpt == -1) & (sigma_h2_map != 0) & (~np.isnan(sigma_h2_map))
    ngood = np.sum(good.astype(int))
    return ngood


data.add_property('n_sf_pix',
                  get_n_sf_pix,
                  property_label=r"$n_\mathrm{pix}$",
                  property_bins=None)

# data.add_property_label('av', r"$A_V$")
pl.close('all')


def slope_intercept_correlations_global(data):
    '''
    Make plots of N vs global property (left column), and log C vs global
    property (right column). Save these plots. Then make a latex table of the
    correlation coefficients (Spearman, Pearson, Kendall tau). Need to copy-paste
    table into paper.
    '''
    def classify_spiral(galname):
        hubble, hubble_min, hubble_max = surfd.get_hubble_type_califa(galname)
        if (hubble == hubble_min) and (hubble == hubble_max) and (hubble == 'S'):
            # Must be absolutely certain about hubble type (S, I or E)
            return True
        else:
            return False


    def classify_starforming_spiral(galname):
        ionclass = get_ionclass(galname)[0]
        spiral = classify_spiral(galname)
        re = get_re_arcsec(galname)
        if ionclass == 'Starforming':
            # Must be absolutely certain about hubble type (S, I or E)
            return (True and spiral
                    and (re > 15))  #(2*np.sqrt(re**2 + 3.3**2) > 6*5))
        else:
            return False


    calc_pearson = dict()
    calc_pearson['S'] = classify_starforming_spiral  #classify_spiral
    marker_hubble_types = {'S': '*'}
    corr_coefs_fixed_alpha = data.plot_sigma_fit_params_vs_other_property(
        figsize=(5, 39),
        which_fit='linmix_fit_reverse',
        which_alpha='fixed',
        marker_hubble_types=marker_hubble_types,
        calc_pearson=calc_pearson)
    pl.tight_layout()
    pl.savefig('n_and_slope_vs_global_all_r_sf_spiral_const_alphaco_allpix.pdf')

    calc_pearson = dict()
    calc_pearson['S'] = classify_starforming_spiral  #classify_spiral
    marker_hubble_types = {'S': '*'}
    corr_coefs_alpha_sf = data.plot_sigma_fit_params_vs_other_property(
        figsize=(5, 39),
        which_fit='fit_result_reverse',
        which_alpha='fixed_sf_only',
        marker_hubble_types=marker_hubble_types,
        calc_pearson=calc_pearson)
    pl.tight_layout()
    pl.savefig('n_and_slope_vs_global_all_r_sf_spiral_const_alphaco_sf_only.pdf')

    calc_pearson = dict()
    calc_pearson['S'] = classify_starforming_spiral  #classify_spiral
    marker_hubble_types = {'S': '*'}
    corr_coefs_alpha_met = data.plot_sigma_fit_params_vs_other_property(
        figsize=(5, 39),
        which_fit='fit_result_reverse',
        which_alpha='met_alpha_sf_only',
        marker_hubble_types=marker_hubble_types,
        calc_pearson=calc_pearson)
    pl.tight_layout()
    pl.savefig('n_and_slope_vs_global_all_r_sf_spiral_metdep_alphaco_sf_only.pdf')

    # Now make a table with the correlation coefficients
    tbl = {
        'Property': [],
        r'$r_\mathrm{P}$ 1': [],
        r'$r_\mathrm{S}$ 1': [],
        r'$\tau_\mathrm{K} 1$': [],
        r'$r_\mathrm{P}$ 2': [],
        r'$r_\mathrm{S}$ 2': [],
        r'$\tau_\mathrm{K} 2$': [],
        r'$r_\mathrm{P}$ 3': [],
        r'$r_\mathrm{S}$ 3': [],
        r'$\tau_\mathrm{K} 3$': [],
        r'$r_\mathrm{P}$ 4': [],
        r'$r_\mathrm{S}$ 4': [],
        r'$\tau_\mathrm{K} 4$': [],
        r'$r_\mathrm{P}$ 5': [],
        r'$r_\mathrm{S}$ 5': [],
        r'$\tau_\mathrm{K} 5$': [],
        r'$r_\mathrm{P}$ 6': [],
        r'$r_\mathrm{S}$ 6': [],
        r'$\tau_\mathrm{K} 6$': []
    }
    #
    # corr_coefs_fixed_alpha
    # corr_coefs_alpha_sf
    # corr_coefs_alpha_met

    for k in corr_coefs_alpha_met['slope'].keys():
        tbl['Property'].append(data.global_property_labels[k])
        # Slopes
        # Fixed alpha
        k2 = 'slope'
        tbl[r'$r_\mathrm{P}$ 1'].append(corr_coefs_fixed_alpha[k2][k]['pearson'][0])
        tbl[r'$r_\mathrm{S}$ 1'].append(corr_coefs_fixed_alpha[k2][k]['spearman'][0])
        tbl[r'$\tau_\mathrm{K} 1$'].append(corr_coefs_fixed_alpha[k2][k]['kendall'][0])
        # SF alpha
        tbl[r'$r_\mathrm{P}$ 2'].append(corr_coefs_alpha_sf[k2][k]['pearson'][0])
        tbl[r'$r_\mathrm{S}$ 2'].append(corr_coefs_alpha_sf[k2][k]['spearman'][0])
        tbl[r'$\tau_\mathrm{K} 2$'].append(corr_coefs_alpha_sf[k2][k]['kendall'][0])
        # Metallicity alpha
        tbl[r'$r_\mathrm{P}$ 3'].append(corr_coefs_alpha_met[k2][k]['pearson'][0])
        tbl[r'$r_\mathrm{S}$ 3'].append(corr_coefs_alpha_met[k2][k]['spearman'][0])
        tbl[r'$\tau_\mathrm{K} 3$'].append(corr_coefs_alpha_met[k2][k]['kendall'][0])

        # Intercepts
        # Fixed alpha
        k2 = 'intercept'
        tbl[r'$r_\mathrm{P}$ 4'].append(corr_coefs_fixed_alpha[k2][k]['pearson'][0])
        tbl[r'$r_\mathrm{S}$ 4'].append(corr_coefs_fixed_alpha[k2][k]['spearman'][0])
        tbl[r'$\tau_\mathrm{K} 4$'].append(corr_coefs_fixed_alpha[k2][k]['kendall'][0])
        # SF alpha
        tbl[r'$r_\mathrm{P}$ 5'].append(corr_coefs_alpha_sf[k2][k]['pearson'][0])
        tbl[r'$r_\mathrm{S}$ 5'].append(corr_coefs_alpha_sf[k2][k]['spearman'][0])
        tbl[r'$\tau_\mathrm{K} 5$'].append(corr_coefs_alpha_sf[k2][k]['kendall'][0])
        # Metallicity alpha
        tbl[r'$r_\mathrm{P}$ 6'].append(corr_coefs_alpha_met[k2][k]['pearson'][0])
        tbl[r'$r_\mathrm{S}$ 6'].append(corr_coefs_alpha_met[k2][k]['spearman'][0])
        tbl[r'$\tau_\mathrm{K} 6$'].append(corr_coefs_alpha_met[k2][k]['kendall'][0])

    ascii.write(tbl,
                '/Users/ryan/Dropbox/mac/wise_w3_vs_co/draft/table1.tex',
                Writer=ascii.Latex,
                latexdict={
                    'preamble': r'\centering',
                    'tabletype': 'table*'
                },
                overwrite=True, formats={
                    r'$r_\mathrm{P}$ 1': r'$%2.2f$',
                    r'$r_\mathrm{S}$ 1': r'$%2.2f$',
                    r'$\tau_\mathrm{K} 1$': r'$%2.2f$',
                    r'$r_\mathrm{P}$ 2': r'$%2.2f$',
                    r'$r_\mathrm{S}$ 2': r'$%2.2f$',
                    r'$\tau_\mathrm{K} 2$': r'$%2.2f$',
                    r'$r_\mathrm{P}$ 3': r'$%2.2f$',
                    r'$r_\mathrm{S}$ 3': r'$%2.2f$',
                    r'$\tau_\mathrm{K} 3$': r'$%2.2f$',
                    r'$r_\mathrm{P}$ 4': r'$%2.2f$',
                    r'$r_\mathrm{S}$ 4': r'$%2.2f$',
                    r'$\tau_\mathrm{K} 4$': r'$%2.2f$',
                    r'$r_\mathrm{P}$ 5': r'$%2.2f$',
                    r'$r_\mathrm{S}$ 5': r'$%2.2f$',
                    r'$\tau_\mathrm{K} 5$': r'$%2.2f$',
                    r'$r_\mathrm{P}$ 6': r'$%2.2f$',
                    r'$r_\mathrm{S}$ 6': r'$%2.2f$',
                    r'$\tau_\mathrm{K} 6$': r'$%2.2f$'
                })

    data = {
        'Name': [],
        '$D_L$': [],
        '$T$': [],
        r'$\beta$': [],
        'Det?': [],
        '$r_\mathrm{ap}$': [],
        '$r_{90}$': [],
        '$1.2r(\mathrm{SNR}=2.0)$': [],
        '$F(r\leq r_\mathrm{ap})$': [],
        '$\log M_\mathrm{dust}(r\leq r_\mathrm{ap})$': [],
        '$F(r\leq r_\mathrm{SNR})$': [],
        '$\log M_\mathrm{dust}(r\leq r_\mathrm{SNR})$': []
    }

    ascii.write(data,
                '/Users/ryan/Dropbox/mac/18B_reduce/measurements_20Sept19.tex',
                Writer=ascii.Latex,
                latexdict={
                    'preamble': r'\begin{center}',
                    'tablefoot': r'\end{center}',
                    'tabletype': 'table*',
                    'units': {
                        '$D_L$': 'Mpc',
                        '$T$': 'K',
                        '$r_\mathrm{ap}$': 'arcsec',
                        '$r_{90}$': 'arcsec',
                        '$1.2r(\mathrm{SNR}=2.0)$': 'arcsec',
                        '$F(r\leq r_\mathrm{ap})$': 'mJy',
                        '$\log M_\mathrm{dust}(r\leq r_\mathrm{ap})$':
                        r'$M_\odot$',
                        '$F(r\leq r_\mathrm{SNR})$': 'mJy',
                        '$\log M_\mathrm{dust}(r\leq r_\mathrm{SNR})$':
                        r'$M_\odot$'
                    }
                },
                overwrite=True)


def plot_maps_and_fits(galname, which_fit='lts_fit_reverse'):
    pl.figure(figsize=(14, 6))
    pl.subplot(241)
    ax = pl.gca()
    data.plot_rgb(galname, ax=ax, subplot=[2, 4, 1])
    pl.title('\n' + galname, fontsize=12)

    ax = pl.subplot(242)
    # ax = pl.gca()
    data.map_sigma_12um(galname, subplot=[2, 4, 2], plot_map=True)
    pl.title(r"$\log \Sigma(12\mu \mathrm{m})$ ($L_\odot$pc$^{-2}$)",
             fontsize=10)

    ax = pl.subplot(243)
    # ax = pl.gca()
    sigma_h2_map = data.map_sigma_h2(galname,
                                     which_alpha='fixed',
                                     subplot=[2, 4, 3],
                                     plot_map=True,
                                     return_map=True)
    pl.title(r"$\log \Sigma(\mathrm{H_2})$ ($M_\odot$pc$^{-2}$)", fontsize=10)

    ax = pl.subplot(244)
    # ax = pl.gca()
    data.map_alpha_co(galname,
                      which_alpha='met_alpha_sf_only',
                      subplot=[2, 4, 4],
                      plot_map=True)
    pl.title(r"$\alpha_\mathrm{CO}[12+\log(\mathrm{O/H})]$", fontsize=10)

    pl.subplot(245)
    ax = pl.gca()
    data.plot_sigma_h2_vs_sigma_12(galname,
                                   ax=ax,
                                   which_fit=which_fit,
                                   which_alpha='met_alpha_sf_only')
    pl.title(r"Met. dependent $\alpha_\mathrm{CO}$", fontsize=10)

    xtext = r"$\log \Sigma (\mathrm{H_2})$ ($M_\odot$pc$^{-2}$)"
    ytext = r"$\log \Sigma(12\mu \mathrm{m})$ ($L_\odot$pc$^{-2}$)"
    if 'reverse' in which_fit:
        ytext = r"$\log \Sigma (\mathrm{H_2})$ ($M_\odot$pc$^{-2}$)"
        xtext = r"$\log \Sigma(12\mu \mathrm{m})$ ($L_\odot$pc$^{-2}$)"
    pl.ylabel(ytext,
              fontsize=10)
    pl.xlabel(xtext,
              fontsize=10)

    # data.plot_sigma_h2_vs_sigma_12('NGC6478', ax=ax, which_fit='fit_result_reverse', which_alpha='met_alpha_sf_only')
    pl.subplot(246)
    ax = pl.gca()
    data.plot_sigma_h2_vs_sigma_12(galname,
                                   ax=ax,
                                   which_fit=which_fit,
                                   which_alpha='fixed_sf_only')
    pl.title(r"Fixed $\alpha_\mathrm{CO}$, SF pixels only", fontsize=10)
    pl.xlabel(xtext,
              fontsize=10)
    #
    # pl.xlabel(r"$\log \Sigma(12\mu \mathrm{m})$ ($L_\odot$pc$^{-2}$)",
    #           fontsize=10)

    pl.subplot(247)
    ax = pl.gca()
    if which_fit == 'fit_result_reverse':
        which_fit = 'linmix_fit_reverse'

    if which_fit == 'fit_result_forward':
        which_fit = 'linmix_fit_forward'
    data.plot_sigma_h2_vs_sigma_12(galname,
                                   ax=ax,
                                   which_fit=which_fit,
                                   which_alpha='fixed')
    pl.title(r"Fixed $\alpha_\mathrm{CO}$, all pixels", fontsize=10)
    pl.xlabel(xtext,
              fontsize=10)
    # pl.xlabel(r"$\log \Sigma(12\mu \mathrm{m})$ ($L_\odot$pc$^{-2}$)",
    #           fontsize=10)

    ax = pl.subplot(248)
    # ax = pl.gca()
    bpt = data.map_bpt(galname,
                       subplot=[2, 4, 8],
                       plot_map=True,
                       return_map=True)
    pl.title(r"BPT", fontsize=10)

    good = (bpt == -1) & (sigma_h2_map != 0) & (~np.isnan(sigma_h2_map))
    ngood = np.sum(good.astype(int))
    print("SFing pixels w/ CO detected: %i" % (ngood, ))

    pl.tight_layout(rect=(0, 0, 1, 0.95))
    return ngood


def make_booklet(which_fit):
    ngood = []
    gnames = []
    # which_fit = 'fit_result_reverse'
    # which_fit = 'fit_result_forward'
    for galname in data.data['name']:
        if len(
                glob.glob(
                    '/Users/ryan/venus/shared_data/califa/DR3-stack/%s/%s_result.pk'
                    % (galname, galname))) == 0:
            print("Nothing exists for this galaxy")
            continue

        gnames.append(galname)
        n = plot_maps_and_fits(galname, which_fit=which_fit)
        ngood.append(n)
        pl.savefig('/Users/ryan/Dropbox/mac/wise_w3_vs_co/figs_v4/%s.pdf' %
                   (galname, ))
        pl.close()

    fwd_rev = ''
    if 'forward' in which_fit:
        fwd_rev = 'ks'
    from PyPDF2 import PdfFileMerger
    for version in [15, 20]:
        merger = PdfFileMerger()
        i = 0
        for gname in gnames:
            n = ngood[i]
            if n >= version:
                pdf = '/Users/ryan/Dropbox/mac/wise_w3_vs_co/figs_v4/%s.pdf'%(gname,)
                merger.append(open(pdf, 'rb'))
            i += 1
        with open(
                '/Users/ryan/Dropbox/mac/wise_w3_vs_co/booklet_multi_alphaco_%ipix_%s.pdf' %
            (version, fwd_rev), 'wb') as fout:
            merger.write(fout)


    # merger = PdfFileMerger()
    # version = 6
    # for pdf in glob.glob('/Users/ryan/Dropbox/mac/wise_w3_vs_co/figs_v4/*.pdf'):
    #     merger.append(open(pdf, 'rb'))
    # with open(
    #         '/Users/ryan/Dropbox/mac/wise_w3_vs_co/booklet_multi_alphaco_v%s.pdf' %
    #     (version, ), 'wb') as fout:
    #     merger.write(fout)

for which_fit in ['fit_result_reverse', 'fit_result_forward']:
    make_booklet(which_fit)

cat = data.get_pixel_catalog()
with open('/Users/ryan/Dropbox/mac/wise_w3_vs_co/pixel_catalog.pk', 'wb') as f:
    pickle.dump(cat, f)
f.close()

# merger = PdfFileMerger()
# version = 6
# for pdf in glob.glob('/Users/ryan/Dropbox/mac/wise_w3_vs_co/figs_v4/*.pdf'):
#     merger.append(open(pdf, 'rb'))
# with open(
#         '/Users/ryan/Dropbox/mac/wise_w3_vs_co/booklet_multi_alphaco_v%s.pdf' %
#     (version, ), 'wb') as fout:
#     merger.write(fout)


colours = ['xkcd:bright blue', 'xkcd:muted green', 'xkcd:goldenrod',
    'xkcd:light red']

pl.figure(figsize=(4, 4))
bpt = cat['bpt']
sf = bpt == -1
inter = bpt == 0
lier = bpt == 1
seyfert = bpt == 2
lbls = ['SF', 'Inter.', 'LIER', 'Sy.']
i = 0
for c in [sf, inter, lier, seyfert]:
    # pl.errorbar(cat['log_sig12'][c], cat['log_sigh2_aco_met'][c], xerr= cat['err_log_sig12'][c], yerr=cat['err_log_sigh2_aco_met'][c],
    pl.errorbar(cat['log_sig12'][c], cat['log_sigh2_aco3p2_all'][c], xerr= cat['err_log_sig12'][c], yerr=cat['err_log_sigh2_aco3p2_all'][c],
                marker='s',
                color='k',
                markeredgecolor='k',
                markerfacecolor=colours[i],
                linewidth=0.5,
                linestyle='none',
                alpha=0.5,
                capsize=2,
                markersize=3,
                ecolor='k',
                elinewidth=0.5,
                markeredgewidth=0.5, label=lbls[i])
    i += 1

pl.xlim(-1,2.5)
pl.ylim(-2, 3.5)
pl.xlabel(r"$\log \Sigma(12\mathrm{\mu m})$ $[ L_\odot$pc$^{-2}]$")
pl.ylabel(r"$\log \Sigma(\mathrm{H_2})$ $[ M_\odot$pc$^{-2}]$")
pl.title(r"Using $\alpha_\mathrm{CO}[12+\log\mathrm{O/H}]$ (SF pixels only)",
         fontsize=10)
pl.tight_layout(rect=(0, 0, 1, .95))





pl.figure(figsize=(4, 4))
galname_list = np.array(data.data['name'])[np.argsort(-data.data['n_sf_pix'])]
n_sf_pix = data.data['n_sf_pix'][np.argsort(-data.data['n_sf_pix'])]
galname_list = galname_list[n_sf_pix < 10]
data.slope_intercept(which_fit='fit_result_reverse',
                     which_alpha='met_alpha_sf_only',
                     galname_list=galname_list,
                     plot_fits=True,
                     perform_fit=False,
                     return_fits=False,
                     markerfacecolor='xkcd:cerulean',
                     label='<10 pixels')

galname_list = np.array(data.data['name'])[np.argsort(-data.data['n_sf_pix'])]
n_sf_pix = data.data['n_sf_pix'][np.argsort(-data.data['n_sf_pix'])]
galname_list = galname_list[n_sf_pix >= 10]
res = data.slope_intercept(which_fit='fit_result_reverse',
                     which_alpha='met_alpha_sf_only',
                     galname_list=galname_list,
                     plot_fits=True,
                     perform_fit=True,
                     perform_which_fit='linmix',
                     return_fits=True)

print("Slope:\n median:%2.2f, mean:%2.2f, std:%2.2f"%(np.median(res['slope']),np.average(res['slope']),np.std(res['slope'])))
print("Intercept:\n median:%2.2f, mean:%2.2f, std:%2.2f"%(np.median(res['intercept']),np.average(res['intercept']),np.std(res['intercept'])))
pl.legend(loc='lower center')
pl.xlabel("Intercept")
pl.ylabel("Slope")
pl.title(r"Using $\alpha_\mathrm{CO}[12+\log\mathrm{O/H}]$ (SF pixels only)",
         fontsize=10)
pl.tight_layout(rect=(0, 0, 1, .95))



pl.savefig(
    '/Users/ryan/Dropbox/mac/wise_w3_vs_co/slope_vs_intercept_metalpha_v4.pdf')

pl.figure(figsize=(4, 4))
galname_list = np.array(data.data['name'])[np.argsort(-data.data['n_sf_pix'])]
n_sf_pix = data.data['n_sf_pix'][np.argsort(-data.data['n_sf_pix'])]
galname_list = galname_list[n_sf_pix < 10]
data.slope_intercept(which_fit='fit_result_reverse',
                     which_alpha='fixed_sf_only',
                     galname_list=galname_list,
                     plot_fits=True,
                     perform_fit=False,
                     return_fits=False,
                     markerfacecolor='xkcd:cerulean',
                     label='<10 pixels')

galname_list = np.array(data.data['name'])[np.argsort(-data.data['n_sf_pix'])]
n_sf_pix = data.data['n_sf_pix'][np.argsort(-data.data['n_sf_pix'])]
galname_list = galname_list[n_sf_pix >= 10]
res = data.slope_intercept(which_fit='fit_result_reverse',
                     which_alpha='fixed_sf_only',
                     galname_list=galname_list,
                     plot_fits=True,
                     perform_fit=True,
                     perform_which_fit='linmix',
                     return_fits=True)

print("Slope:\n median:%2.2f, mean:%2.2f, std:%2.2f"%(np.median(res['slope']),np.average(res['slope']),np.std(res['slope'])))
print("Intercept:\n median:%2.2f, mean:%2.2f, std:%2.2f"%(np.median(res['intercept']),np.average(res['intercept']),np.std(res['intercept'])))

pl.legend(loc='lower center')
pl.xlabel("Intercept")
pl.ylabel("Slope")
pl.title(r"Using $\alpha_\mathrm{CO}=3.2$ (SF pixels only)", fontsize=10)
pl.tight_layout(rect=(0, 0, 1, .95))
pl.savefig(
    '/Users/ryan/Dropbox/mac/wise_w3_vs_co/slope_vs_intercept_sfonly_v4.pdf')

pl.figure(figsize=(4, 4))
galname_list = np.array(data.data['name'])[np.argsort(-data.data['n_sf_pix'])]
n_sf_pix = data.data['n_sf_pix'][np.argsort(-data.data['n_sf_pix'])]
galname_list = galname_list[n_sf_pix < 10]
data.slope_intercept(which_fit='linmix_fit_reverse',
                     which_alpha='fixed',
                     galname_list=galname_list,
                     plot_fits=True,
                     perform_fit=False,
                     return_fits=False,
                     markerfacecolor='xkcd:cerulean',
                     label='<10 pixels')

galname_list = np.array(data.data['name'])[np.argsort(-data.data['n_sf_pix'])]
n_sf_pix = data.data['n_sf_pix'][np.argsort(-data.data['n_sf_pix'])]
galname_list = galname_list[n_sf_pix >= 10]
res = data.slope_intercept(which_fit='linmix_fit_reverse',
                     which_alpha='fixed',
                     galname_list=galname_list,
                     plot_fits=True,
                     perform_fit=True,
                     perform_which_fit='linmix',
                     return_fits=True)

print("Slope:\n median:%2.2f, mean:%2.2f, std:%2.2f"%(np.median(res['slope']),np.average(res['slope']),np.std(res['slope'])))
print("Intercept:\n median:%2.2f, mean:%2.2f, std:%2.2f"%(np.median(res['intercept']),np.average(res['intercept']),np.std(res['intercept'])))

pl.legend(loc='upper left')
pl.xlabel("Intercept")
pl.ylabel("Slope")
pl.title(r"Using $\alpha_\mathrm{CO}=3.2$ (all pixels)", fontsize=10)
pl.tight_layout(rect=(0, 0, 1, .95))
pl.ylim(-1, 3.5)
pl.xlim(-2.2, 1.2)
pl.savefig(
    '/Users/ryan/Dropbox/mac/wise_w3_vs_co/slope_vs_intercept_all_v4.pdf')

# data.slope_intercept(which_fit='lts_fit_reverse', which_alpha='met_alpha_sf_only', galname_list=galname_list, plot_fits=True, perform_fit=False)

# data.plot_sigma_h2_vs_sigma_12('NGC6478', ax=ax, which_fit='fit_result_reverse', which_alpha='fixed_sf_only')

for global_property in data.global_properties:
    data.get_fit_params_binned_by_property(global_property)
# data.plot_data_fits('logm_global', 1, 2, 1, legend=False)

data.plot_data_fits('logm_global', 9, 2, 1, legend=False)
data.plot_data_fits('sfr_global', 9, 2, 3, legend=False)
data.plot_data_fits('met_global', 9, 2, 5, legend=False)
data.plot_data_fits('ur_global', 9, 2, 7, legend=False)

data.plot_data_fits('av', 9, 2, 9, legend=False)
data.plot_data_fits('l_tir', 9, 2, 11, legend=False)
data.plot_data_fits('l_fuv', 9, 2, 13, legend=False)
data.plot_data_fits('l_nuv', 9, 2, 15, legend=False)
data.plot_data_fits('l_w4', 9, 2, 17, legend=False)

# b/a correlates with sersic
# check simard, gama,... for sersic
from ltsfit.lts_linefit import lts_linefit
good = (data.data['interacting'] == 'I') & (data.data['dont_fit'] == False)

data.data['name'][good]
x = data.data['slope'][good]
y = data.data['intercept'][good]
xerr = data.data['slope_err'][good]
yerr = data.data['intercept_err'][good]
lts_linefit(x, y, xerr, yerr)
surfd.scatterplot(x, y, 'slope', 'intercept', xerr=xerr, yerr=yerr, ecolor='b')
print('Spearman r=%.2g and p=%.2g' % scipy.stats.spearmanr(x, y))
print('Pearson r=%.2g and p=%.2g' % scipy.stats.pearsonr(
    data.data['slope'][good], data.data['intercept'][good]))
good = (data.data['interacting'] == 'M') & (data.data['dont_fit'] == False)
surfd.scatterplot(data.data['slope'][good],
                  data.data['intercept'][good],
                  'slope',
                  'intercept',
                  xerr=data.data['slope_err'][good],
                  yerr=data.data['intercept_err'][good],
                  ecolor='r',
                  label='Merger/pair')

import sklearn
from sklearn.decomposition import PCA

# data = plot_sigma_fit_params_vs_global_logmstar(return_data=True)
X = data.data
# , data['met_global']
good = (X['slope_err'] < .75) & (X['dont_fit'] == False) & (
    X['interacting'] == 'I') & (~np.isnan(X['met_global'])) & (
        ~np.isnan(X['l_w4'])) & (X['intercept_err'] < 2)

n_components = 3
pca_data = np.vstack([
    data.data['intercept'], data.data['logm_global'], data.data['ur_global'],
    data.data['l_w4']
]).T
pca_data = np.vstack(
    [data.data['intercept'], data.data['logm_global'],
     data.data['ur_global']]).T
pca_data = pca_data[good, :]
pca_data_av = np.average(pca_data, axis=0)
pca_data -= pca_data_av
pca = PCA(n_components=n_components)
pca.fit(pca_data)
print(pca.explained_variance_ratio_)
pc = pca.components_[n_components - 1]

n_properties = len(data.global_properties)
X = data.data
found = False
best_coeff = 0
best_var = 0
for n in range(1, n_properties):
    # if found == True:
    #     break
    print("n = %i" % (n, ))
    combos_temp = list(itertools.combinations(np.arange(n_properties), n))
    n_components = n + 1
    # For slope
    for combo in combos_temp:
        properties_temp = np.array(data.global_properties)[list(combo)]
        print(properties_temp)
        prop_data = [data.data[s] for s in properties_temp]
        prop_data = [data.data['intercept']] + prop_data
        pca_data = np.vstack(prop_data).T
        good = (~np.isnan(pca_data[:, 0])) & (~np.isinf(pca_data[:, 0])) & (
            X['slope_err'] < .75) & (X['dont_fit'] == False) & (
                X['interacting'] == 'I') & (X['intercept_err'] < 2)
        for i in range(1, pca_data.shape[1]):
            good = good & (~np.isnan(pca_data[:, i])) & (
                ~np.isinf(pca_data[:, i]))
        pca_data = pca_data[good, :]
        pca_data_av = np.average(pca_data, axis=0)
        pca_data -= pca_data_av
        pca = PCA(n_components=n_components)
        pca.fit(pca_data)
        print(pca.explained_variance_ratio_)
        pc = pca.components_[n_components - 1]
        print(pc)
        if (abs(pc[0]) > 0.1) & (pca.explained_variance_ratio_[-1] < 0.001):
            print("GOOD COMBO FOUND")
            if abs(pc[0]) > best_coeff:
                best_coeff = abs(pc[0])
                # found = True
                best_pc = pc
                best_pca = pca.explained_variance_ratio_
                best_params = properties_temp
                best_data = pca_data
                best_av = pca_data_av
                # break
        print("==")

best_intercept = [
    best_coeff, best_pc, best_pca, best_params, best_data, best_av
]

n_properties = len(data.global_properties)
X = data.data
found = False
best_coeff = 0
best_var = 0
for n in range(1, n_properties):
    # if found == True:
    #     break
    print("n = %i" % (n, ))
    combos_temp = list(itertools.combinations(np.arange(n_properties), n))
    n_components = n + 1
    # For slope
    for combo in combos_temp:
        properties_temp = np.array(data.global_properties)[list(combo)]
        print(properties_temp)
        prop_data = [data.data[s] for s in properties_temp]
        prop_data = [data.data['slope']] + prop_data
        pca_data = np.vstack(prop_data).T
        good = (~np.isnan(pca_data[:, 0])) & (~np.isinf(pca_data[:, 0])) & (
            X['slope_err'] < .75) & (X['dont_fit'] == False) & (
                X['interacting'] == 'I') & (X['intercept_err'] < 2)
        for i in range(1, pca_data.shape[1]):
            good = good & (~np.isnan(pca_data[:, i])) & (
                ~np.isinf(pca_data[:, i]))
        pca_data = pca_data[good, :]
        pca_data_av = np.average(pca_data, axis=0)
        pca_data -= pca_data_av
        pca = PCA(n_components=n_components)
        pca.fit(pca_data)
        print(pca.explained_variance_ratio_)
        pc = pca.components_[n_components - 1]
        print(pc)
        if (abs(pc[0]) > 0.15) & (pca.explained_variance_ratio_[-1] < 0.001):
            print("GOOD COMBO FOUND")
            if abs(pc[0]) > best_coeff:
                best_coeff = abs(pc[0])
                # found = True
                best_pc = pc
                best_pca = pca.explained_variance_ratio_
                best_params = properties_temp
                best_data = pca_data
                best_av = pca_data_av
                # break
        print("==")

best_slope = [best_coeff, best_pc, best_pca, best_params, best_data, best_av]

best_coeff, best_pc, best_pca, best_params, best_data, best_av = best_intercept

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
RANDOM_STATE = 42

n_components = 3
features = np.vstack(
    [data.data[s] for s in ['logm_global', 'ur_global', 'v_disp_ha']]).T

features = np.vstack([data.data['slope']] + [
    data.data[s] for s in
    ['v_disp_stellar', 'v_disp_ha', 'dmpc', 'ba_g_disk', 'logm_global']
]).T
n_components = features.shape[1]
good_rows = []
for i in range(0, features.shape[0]):
    if any(np.isnan(features[i])) or any(np.isinf(features[i])):
        continue
    else:
        good_rows.append(i)

target = data.data['slope'][good_rows]
features = features[good_rows, :]

scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
pca = PCA(n_components=n_components)
pca.fit(features)
print(pca.explained_variance_ratio_)
pc = pca.components_[n_components - 1]
print(pc)

pl.scatter(
    -(features[:, 1] * pc[1] + features[:, 2] * pc[2] + features[:, 3] * pc[3]
      + features[:, 4] * pc[4] + features[:, 5] * pc[5]) / pc[0],
    features[:, 0])

# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.30,
                                                    random_state=RANDOM_STATE)

std_clf = make_pipeline(StandardScaler(), PCA(n_components=n_components),
                        LinearRegression())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)
mse = mean_squared_error(std_clf.predict(X_test), y_test)
pca_std = std_clf.named_steps['pca']
print('\nPC 1 with scaling:\n', pca_std.components_[0])
scaler = std_clf.named_steps['standardscaler']
X_train_std_transformed = pca_std.transform(scaler.transform(X_train))

from sklearn.model_selection import cross_val_score
std_clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(std_clf, X_train_std_transformed, y_train, cv=5)

pl.figure()
yy = (best_data + best_av)[:, 0]
xx = -((best_data + best_av)[:, 1] * best_pc[1] +
       (best_data + best_av)[:, 2] * best_pc[2] +
       (best_data + best_av)[:, 3] * best_pc[3] +
       (best_data + best_av)[:, 4] * best_pc[4] +
       (best_data + best_av)[:, 5] * best_pc[5] +
       (best_data + best_av)[:, 6] * best_pc[6] +
       (best_data + best_av)[:, 7] * best_pc[7] +
       (best_data + best_av)[:, 8] * best_pc[8] +
       (best_data + best_av)[:, 9] * best_pc[9] -
       ((best_av)[1] * best_pc[1] + (best_av)[2] * best_pc[2] +
        (best_av)[3] * best_pc[3] + (best_av)[4] * best_pc[4] +
        (best_av)[5] * best_pc[5] + (best_av)[6] * best_pc[6] +
        (best_av)[7] * best_pc[7] + (best_av)[8] * best_pc[8] +
        (best_av)[9] * best_pc[9])) / best_pc[0]

pl.scatter(xx, yy - xx)
#
#     # For intercept
# In [142]: best_pc
# Out[142]: array([-0.38829384, -0.34161445,  0.63421943,  0.57470791,  0.00199907])
#
# In [143]: best_pca
# Out[143]: array([0.82166349, 0.07216508, 0.05977481, 0.03640044, 0.00999619])
#
# In [144]: best_params
# Out[144]: array(['sfr_global', 'ur_global', 'l_fuv', 'n_g'], dtype='<U11')
#
# ['ur_global' 'l_tir' 'l_nuv' 'n_g']
# [0.83399816 0.06125778 0.05361592 0.04194401 0.00918413]
# [-0.19632894 -0.5476484   0.42253234 -0.69483845 -0.01421731]
