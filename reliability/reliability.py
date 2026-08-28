import numpy as np
import pandas as pd
import os
import astropy
from astropy.io import fits
from glob import glob
#import VBMicrolensing
import RTModel
#import jacscanomaly
#import pyAethra
from pyAethra import load_config, load_and_run
from jacscanomaly import CandidateCriteria, Finder, FinderConfig

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 1})
pylab_params = {'legend.fontsize': 'large',
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'large',
    'ytick.labelsize': 'large'}
pylab.rcParams.update(pylab_params)

path = "/Users/chrislam/Desktop/birp/"
data_path = "/Users/chrislam/Desktop/microlensing/data/false_positives/"

f146_times = np.load(data_path + 'roman_times_shortcadence.npy')
f087_times = np.load(data_path + 'roman_times_longcadence.npy')
f213_times = np.load(data_path + 'roman_times_longcadence2.npy')
f146_times = f146_times - 2450000
f087_times = f087_times - 2450000
f213_times = f213_times - 2450000

# Type 2 Cepheid variables, from Improved_Noise folder
t2cep_path = data_path + 'RGES_filters_T2CEP_lightcurves_final/' # 100 files
t2cep_path_processed = data_path + 'processed/RGES_filters_T2CEP_lightcurves_final/'
t2cep_files = sorted(glob(t2cep_path+'*'))

# M dwarf flares, from Simple_Noise folder
fl_path = data_path + 'RGES_filters_FL_lightcurves/RGES_filters_FL_lightcurves_final/'
fl_path_processed = data_path + 'processed/RGES_filters_FL_lightcurves/RGES_filters_FL_lightcurves_final/'
fl_files = sorted(glob(fl_path+'*'))

"""
for i in range(len(fl_files)):
    fits_image_filename = fl_files[i]
    # strip preamble and suffix from file name
    stripped_name = fits_image_filename.removeprefix(fl_path+'RGES_filters_')
    stripped_name = stripped_name.removesuffix('_lightcurves.fits')

    event_path = fl_path_processed+stripped_name+'/'
    event_data_path = fl_path_processed+stripped_name+'/Data/'
    try:
        os.mkdir(event_path)
        print(f"Directory '{event_path}' created")
    except:
        print(f"Directory '{event_path}' already exists")
    try:
        os.mkdir(event_data_path)
        print(f"Directory '{event_data_path}' created")
    except:
        print(f"Directory '{event_data_path}' already exists")

    # open fits data
    hdul = fits.open(fits_image_filename)
    # print(hdul[0].header)
    # print(hdul[1].header)
    # print(hdul[2].header)
    # print(hdul[3].header)
    # print(hdul[1].data)
    # print(hdul[2].data)
    # print(hdul[3].data)
    f087_y, f087_yerr = zip(*hdul[1].data)
    f146_y, f146_yerr = zip(*hdul[2].data)
    f213_y, f213_yerr = zip(*hdul[3].data)

    df_f087 = pd.DataFrame({'# Mag': f087_y, 'err': f087_yerr, 'HJD-2450000': f087_times})
    df_f146 = pd.DataFrame({'# Mag': f146_y, 'err': f146_yerr, 'HJD-2450000': f146_times})
    df_f213 = pd.DataFrame({'# Mag': f213_y, 'err': f213_yerr, 'HJD-2450000': f213_times})
    df_f087.to_csv(event_data_path+'w087sat3.dat', sep=' ', index=False)
    df_f146.to_csv(event_data_path+'w146sat1.dat', sep=' ', index=False)
    df_f213.to_csv(event_data_path+'w213sat2.dat', sep=' ', index=False)

    rtm = RTModel.RTModel(event_path)
    rtm.set_satellite_dir('/satellitedir')
    rtm.config_InitCond(npeaks=5, nostatic = False, usesatellite = 1)
    rtm.run()
    quit()

    plt.errorbar(f146_times, f146_y, f146_yerr, label='F146')
    plt.errorbar(f087_times, f087_y, f087_yerr, label='F087')
    plt.errorbar(f213_times, f213_y, f213_yerr, label='F213')
    plt.xlabel('time [BJD-2450000]')
    plt.ylabel('magnitude')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
    quit()

    ### fit binary lens event
    #VBM = VBMicrolensing.VBMicrolensing()
    rtm = RTModel.RTModel()
    s = 0.9       # Separation between the lenses
    q = 0.1       # Mass ratio
    u0 = 0.0       # Impact parameter with respect to center of mass
    alpha = 1.0       # Angle of the source trajectory
    rho = 0.01       # Source radius
    tE = 1.0      # Einstein time in days
    t0 = 1663.45      # Time of closest approach to center of mass

    # Array of parameters. Note that s, q, rho and tE are in log-scale
    pr = [math.log(s), math.log(q), u0, alpha, math.log(rho), math.log(tE), t0]

    #t = np.linspace(t0-tE, t0+tE, 300) # Array of times
    t = f146_times

    magnifications, y1, y2 = VBM.BinaryLightCurve(pr,t)      # Calculation of binary-lens light curve
    plt.plot(t, magnifications)
    plt.show()
    quit()
"""

# create global table columns
labels = []
t0s = []
tEs = []
u0s = []
t0_anomalies = []
jacscanomaly_scores = []
for i in range(len(t2cep_files)):
    fits_image_filename = t2cep_files[i] 

    # grab event name
    name = fits_image_filename.removeprefix(t2cep_path+'RGES_filters_OGLE-BLG-')
    name = name.removesuffix('_lightcurves_final.fits')

    hdul = fits.open(fits_image_filename)
    # print(hdul[1].header)
    # print(hdul[2].header)
    # print(hdul[3].header)
    # print(hdul[1].data)
    # print(hdul[2].data)
    # print(hdul[3].data)

    f087_times, f087_y, f087_yerr  = zip(*hdul[1].data)
    f146_times, f146_y, f146_yerr = zip(*hdul[2].data)
    f213_times, f213_y, f213_yerr = zip(*hdul[3].data)
    f087_times = np.array(f087_times) - 2450000
    f146_times = np.array(f146_times) - 2450000
    f213_times = np.array(f213_times) - 2450000

    # visualize
    # plt.errorbar(f146_times, f146_y, f146_yerr, label='F146')
    # plt.errorbar(f087_times, f087_y, f087_yerr, label='F087')
    # plt.errorbar(f213_times, f213_y, f213_yerr, label='F213')
    # plt.xlabel('time [BJD-2450000]')
    # plt.ylabel('magnitude')
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.show()

    # stack to aethra-preferred format, which wants different rows per filter, rather than different columns
    f146_df = pd.DataFrame({'name': name, 'bjd': f146_times, 'mag': f146_y, 'mag_err': f146_yerr})
    f087_df = pd.DataFrame({'name': name, 'bjd': f087_times, 'mag': f087_y, 'mag_err': f087_yerr})
    f213_df = pd.DataFrame({'name': name, 'bjd': f213_times, 'mag': f213_y, 'mag_err': f213_yerr})
    lc_data = pd.concat([f146_df, f087_df, f213_df], ignore_index=True)

    ### EVENT DETECTION WITH PYAETHRA
    aethra_config = load_config(path+"config/pyAethra_config.yaml") # courtesy of Katie Vandorou; DO NOT CHANGE FILE CONTENTS
    aethra_results_df = load_and_run(lc_data, aethra_config)
    label = aethra_results_df['label'].iloc[0]
    labels.append(label)
    print("label: ", label)

    ### this is logic for proceeding to next event in the loop if aethra fails to find a bound or free-floating planet candidate
    if (label != 'microlensing') and (label != 'ffp'):
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        continue # skip to next event 

    t0_fit = aethra_results_df['t0_fit'].iloc[0]
    u0_fit = aethra_results_df['u0_fit'].iloc[0]
    tE_fit = aethra_results_df['tE_fit'].iloc[0]
    print("t0: ", t0_fit)
    print("u0: ", u0_fit)
    print("tE: ", tE_fit)
    t0s.append(t0_fit)
    tEs.append(tE_fit)
    u0s.append(u0_fit)

    ### ANOMALY DETECTION WITH JACSCANOMALY
    time = np.array(f146_df["bjd"])
    mag = np.array(f146_df["mag"])
    magerr = np.array(f146_df["mag_err"])

    criteria = CandidateCriteria(
        min_dchi2=20.0, 
        min_n_eff=2.0,
        #min_n_contrib=2,
        #max_peak_frac=0.8,
    )
    jacscanomaly_config = FinderConfig(
        fitter_kind="pspl",
        candidate_criteria=criteria, # candidate_criteria=CandidateCriteria(min_n_eff=2.0)
        gap=150.0 # default is 50. gap between seasons is ~half a year, hence 150
    )
    finder = Finder(jacscanomaly_config)

    # initial guess is from aethra
    p0 = np.array([t0_fit, tE_fit, u0_fit]) 

    # actually run the anomaly finder. Make sure you're distinguishing between magnitude and flux. Beginner tier uses mag; experienced+ tiers use flux.
    anomaly_result = finder.run(time, mag, magerr, p0, data_kind="mag")
    #anomaly_result.print_summary()
    anomaly_result_summary = anomaly_result.summary_table()
    print(anomaly_result_summary)

    # grab relevant variables
    t0_fit_best = anomaly_result_summary['best_t0'].values[0]
    print("t0_fit_best: ", t0_fit_best)

    best_score = anomaly_result_summary['best_score'].values[0]
    print("best score: ", best_score)
    jacscanomaly_scores.append(best_score)

    # this is where I skip to the next event, if jacscanomaly finds nothing
    if best_score < 10:
        print("does not pass score threshold of 10")
        t0_anomalies.append(np.nan)
        continue
    t0_anomaly = anomaly_result_summary['best_t0'].values[0]
    t0_anomalies.append(t0_anomaly)

    # EVENT MODELING WITH RTMODEL
    # maybe do this separately

result_df = pd.DataFrame({'name': name, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(result_df)
