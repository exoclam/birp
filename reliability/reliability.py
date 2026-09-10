"""
Copy pasted from Alex Stephan's WG6 improved_noise folder README.

A given fits-file contains the resampled and filter-translated lightcurves of a variable
source. The original data was taken from OGLE and UKIRT, then resampled with Gaussian
process fitting, and finally interpolated to match OGLE V and I filters and UKIRT H and K
filters with the Roman filters planned to be used for the microlensing survey. An exception are flares, which are taken from TESS data.
Additionally, uncertainties are applied based on estimated signal-to-noise levels for Roman. 
The Roman filters are: 
F146 (wide filter with H band overlap) on a short cadence (about 12.1 minutes)
F087 (bluer filter with I band overlap) on a long cadence (about 6 hours)
F213 (redder filter with K band overlap) on a long cadence (about 6 hours)
F087 and F213 are planned to be offset by 3 hours. 
Timestamps for the lightcurves are included in the fits-files.

The added noise includes both photometric noise as well as noise from blending/crowding.

As the OGLE and UKIRT data is often too bright for Roman’s sensitivity range, the lightcurves have been shifted in magnitude to fall within the 15th to 24th magnitude range.

Long Period Variables (LPV)
RR Lyraes (RRLYR)
Heartbeat Stars (HB)
Eclipsing Binaries (ECL)
Ellipsoidal Binaries (ELL)
Delta Scutis (DSCT)
Cepheids, type 1 and 2 (CEP, T2CEP)
M-dwarf flares with rotational variability (FL)

Addition of more variability types is in progress. 
"""

import numpy as np
import pandas as pd
import os
import astropy
from astropy.io import fits
from glob import glob
from tqdm import tqdm
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

### READ IN DATA
f146_times = np.load(data_path + 'roman_times_shortcadence.npy')
f087_times = np.load(data_path + 'roman_times_longcadence.npy')
f213_times = np.load(data_path + 'roman_times_longcadence2.npy')
# f146_times = f146_times - 2450000
# f087_times = f087_times - 2450000
# f213_times = f213_times - 2450000

# Type 1 Cepheid variables, from Improved_Noise folder
cep_path = data_path + 'RGES_filters_CEP_lightcurves_final/' 
cep_files = sorted(glob(cep_path+'*'))

# Delta Scutis, from Improved_Noise folder
dsct_path = data_path + 'RGES_filters_DSCT_lightcurves_final/'
dsct_files = sorted(glob(dsct_path+'*'))

# eclipsing binaries, from Improved_Noise folder
ecl_path = data_path + 'RGES_filters_ECL_lightcurves_final/' # 100 files
ecl_files = sorted(glob(ecl_path+'*'))

# ellipsoidal binaries, from Improved_Noise folder
ell_path = data_path + 'RGES_filters_ELL_lightcurves_final/' 
ell_files = sorted(glob(ell_path+'*'))

# heartbeat stars, from Improved_Noise folder
hb_path = data_path + 'RGES_filters_HB_lightcurves_final/'
hb_files = sorted(glob(hb_path+'*'))

# long period variables, from Improved_Noise folder
lpv_path = data_path + 'RGES_filters_LPV_lightcurves_final/'
lpv_files = sorted(glob(lpv_path+'*'))

# RR Lyraes, from Improved_Noise folder
rrlyr_path = data_path + 'RGES_filters_RRLYR_lightcurves_final/'
rrlyr_files = sorted(glob(rrlyr_path+'*'))

# Type 2 Cepheid variables, from Improved_Noise folder
t2cep_path = data_path + 'RGES_filters_T2CEP_lightcurves_final/' # 100 files
t2cep_files = sorted(glob(t2cep_path+'*'))

# M dwarf flares, from Improved_Noise folder, first half because folder is too big
fl1_path = data_path + 'RGES_filters_FL_to_1900_160329609_284_WFmag_21_lightcurves_final/'
fl1_files = sorted(glob(fl1_path+'*'))
fl2_path = data_path + 'RGES_filters_FL_to_1999_587_flat_WFmag_21_lightcurves_final/'
fl2_files = sorted(glob(fl2_path+'*'))

# M dwarf flares, from Simple_Noise folder
#fl_path = data_path + 'RGES_filters_FL_lightcurves/RGES_filters_FL_lightcurves_final/'
#fl_files = sorted(glob(fl_path+'*'))

def process_event(fits_path, fits_image_filename, names, noise_flag='improved'):
    """Process event light curve

    Args:
        fits_path (str): e.g., t2cep_path
        fits_image_filename (str): element of list like, e.g., t2cep_files
        noise_flag (str): 'simple' or 'improved'
        names (list): list o' names

    Returns:
        lc_data: Pandas DataFrame of name | bjd-2450000 | mag | mag_err, split by filter, ready for pyAethra
        names (list): list o' names, plus this one
        f146_df: Pandas DataFrame; lc_data, but only for F146 filter, for jacscanomaly
    """

    # grab event name
    name = fits_image_filename.removeprefix(fits_path+'RGES_filters_')
    name = name.removesuffix('_lightcurves_final.fits')
    names.append(name)

    hdul = fits.open(fits_image_filename)
    f087_times, f087_y, f087_yerr  = zip(*hdul[1].data)
    f146_times, f146_y, f146_yerr = zip(*hdul[2].data)
    f213_times, f213_y, f213_yerr = zip(*hdul[3].data)
    f087_times = np.array(f087_times) #- 2450000
    f146_times = np.array(f146_times) #- 2450000
    f213_times = np.array(f213_times) #- 2450000

    # stack to aethra-preferred format, which wants different rows per filter, rather than different columns
    f146_df = pd.DataFrame({'name': name, 'bjd': f146_times, 'mag': f146_y, 'mag_err': f146_yerr})
    f087_df = pd.DataFrame({'name': name, 'bjd': f087_times, 'mag': f087_y, 'mag_err': f087_yerr})
    f213_df = pd.DataFrame({'name': name, 'bjd': f213_times, 'mag': f213_y, 'mag_err': f213_yerr})

    # drop NaNs
    f146_df = f146_df.dropna()
    f087_df = f087_df.dropna()
    f213_df = f213_df.dropna()
    
    lc_data = pd.concat([f146_df, f087_df, f213_df], ignore_index=True)

    return lc_data, names, f146_df

def detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores):#, ss, qs, alphas, rhos):

    ### EVENT DETECTION WITH PYAETHRA
    aethra_results_df = load_and_run(lc_data, aethra_config)
    label = aethra_results_df['label'].iloc[0]
    labels.append(label)
    print("label: ", label)

    ### this is logic for proceeding to next event in the loop if pyAethra fails to find a bound or free-floating planet candidate
    if (label != 'microlensing') and (label != 'ffp'):
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        #continue # skip to next event 
        t0_anomalies.append(np.nan)
        jacscanomaly_scores.append(np.nan)
        # ss.append(np.nan)
        # qs.append(np.nan)
        # alphas.append(np.nan)
        # rhos.append(np.nan)

    else:
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
        print("running jacscanomaly...")
        anomaly_result = finder.run(time, mag, magerr, p0, data_kind="mag")
        #anomaly_result.print_summary()
        anomaly_result_summary = anomaly_result.summary_table()
        #print(anomaly_result_summary)

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
        else:
            t0_anomaly = anomaly_result_summary['best_t0'].values[0]
            t0_anomalies.append(t0_anomaly)

    return labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores

### ANALYZE RELIABILITY RUNS
# read in runs
df_fl1 = pd.read_csv(path+'data/reliability_wg6_fl1.csv')
df_fl2 = pd.read_csv(path+'data/reliability_wg6_fl2.csv')
df_ecl = pd.read_csv(path+'data/reliability_wg6_ecl.csv')
df_ell = pd.read_csv(path+'data/reliability_wg6_ell.csv')
df_hb = pd.read_csv(path+'data/reliability_wg6_hb.csv')
df_lpv = pd.read_csv(path+'data/reliability_wg6_lpv.csv') 
df_dsct = pd.read_csv(path+'data/reliability_wg6_dsct.csv')
df_rrlyr = pd.read_csv(path+'data/reliability_wg6_rrlyr.csv')
df_cep = pd.read_csv(path+'data/reliability_wg6_cep.csv')
df_t2cep = pd.read_csv(path+'data/reliability_wg6_t2cep.csv')

# add column denoting what the injected flavor of variability is
df_fl1['variability_flavor'] = df_fl1['name'].str.split('_').str[0]
df_fl2['variability_flavor'] = df_fl2['name'].str.split('_').str[0]
df_ecl['variability_flavor'] = df_ecl['name'].str.split('-').str[2]
df_ell['variability_flavor'] = df_ell['name'].str.split('-').str[2]
df_hb['variability_flavor'] = df_hb['name'].str.split('-').str[2]
df_lpv['variability_flavor'] = df_lpv['name'].str.split('-').str[2]
df_dsct['variability_flavor'] = df_dsct['name'].str.split('-').str[2]
df_rrlyr['variability_flavor'] = df_rrlyr['name'].str.split('-').str[2]
df_cep['variability_flavor'] = df_cep['name'].str.split('-').str[2]
df_t2cep['variability_flavor'] = df_t2cep['name'].str.split('-').str[2]
df = pd.concat([df_fl1, df_fl2, df_ecl, df_ell, df_hb, df_lpv, df_dsct, df_rrlyr, df_cep, df_t2cep], ignore_index=True)
print(df)

print("number of flares: ", len(df_fl1)+len(df_fl2))
print("number of eclipsing binaries: ", len(df_ecl))
print("number of ellipsoidal binaries: ", len(df_ell))
print("number of heartbeat stars: ", len(df_hb))
print("number of long-period variables: ", len(df_lpv))
print("number of Delta Scutis: ", len(df_dsct))
print("number of RR Lyrae stars: ", len(df_rrlyr))
print("number of Cepheid variables: ", len(df_cep))
print("number of Type II Cepheids: ", len(df_t2cep))
df.to_csv(path+'data/reliability_wg6.csv', index=False)

print("")
print("number of events with microlensing or free-floating planet label: ", len(df.loc[(df.label=='microlensing') | (df.label=='ffp')]))
print("fraction of events with microlensing or free-floating planet label: ", len(df.loc[(df.label=='microlensing') | (df.label=='ffp')])/len(df))
print("number of events with microlensing label: ", len(df.loc[df.label=='microlensing']))
print("fraction of events with microlensing label: ", len(df.loc[df.label=='microlensing'])/len(df))
print("number of events with free-floating planet label: ", len(df.loc[df.label=='ffp']))
print("fraction of events with free-floating planet label: ", len(df.loc[df.label=='ffp'])/len(df))

microlensing = df.loc[df.label=='microlensing']
microlensing_score = microlensing.loc[microlensing.jacscanomaly_score>10]
print("")
print("number of events mistaken for microlensing: ", len(microlensing_score))
print("fraction of events mistaken for microlensing: ", len(microlensing_score)/len(df))
print("number of flare events mistaken for microlensing: ", len(microlensing_score.loc[microlensing_score.variability_flavor=='FL']))
print("fraction of flare events mistaken for microlensing: ", len(microlensing_score.loc[microlensing_score.variability_flavor=='FL'])/(len(df_fl1)+len(df_fl2)))
print("fraction of eclipsing binary events mistaken for microlensing: ", len(microlensing_score.loc[microlensing_score.variability_flavor=='ECL'])/len(df_ecl))
print("fraction of ellipsoidal binary events mistaken for microlensing: ", len(microlensing_score.loc[microlensing_score.variability_flavor=='ELL'])/len(df_ell))
print("fraction of heartbeat star events mistaken for microlensing: ", len(microlensing_score.loc[microlensing_score.variability_flavor=='HB'])/len(df_hb))
print("fraction of long-period variable events mistaken for microlensing: ", len(microlensing_score.loc[microlensing_score.variability_flavor=='LPV'])/len(df_lpv))
print("fraction of Delta Scuti events mistaken for microlensing: ", len(microlensing_score.loc[microlensing_score.variability_flavor=='DSC'])/len(df_dsct))
print("fraction of RR Lyrae events mistaken for microlensing: ", len(microlensing_score.loc[microlensing_score.variability_flavor=='RRLYR'])/len(df_rrlyr))
print("fraction of Cepheid variable events mistaken for microlensing: ", len(microlensing_score.loc[microlensing_score.variability_flavor=='CEP'])/len(df_cep))
print("fraction of Type II Cepheid events mistaken for microlensing: ", len(microlensing_score.loc[microlensing_score.variability_flavor=='T2CEP'])/len(df_t2cep))


### RUN THIS ONCE, THEN NEVER AGAIN
### GLOBAL FILES, VARIABLES, AND SETTINGS
aethra_config = load_config(path+"config/pyAethra_config.yaml") # courtesy of Katie Vandorou; DO NOT CHANGE FILE CONTENTS


### BEGIN WORKFLOW
# create global table columns
names = []
labels = []
t0s = []
tEs = []
u0s = []
t0_anomalies = []
jacscanomaly_scores = []
ss = []
qs = []
alphas = []
rhos = []

# inspect a false positive flare 
fp_file = fl1_path+'RGES_filters_'+'FL_0_388857263_504_WFmag_20'+'_lightcurves_final.fits'
fp_lc_data, fp_names, fp_f146_df = process_event(fl1_path, fp_file, [], noise_flag='improved')
print(fp_lc_data)
#labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(fp_lc_data, fp_f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)
window = fp_f146_df.loc[fp_f146_df.bjd.between(2.462184e6 - 0.5*99.963471, 2.462184e6 + 0.5*99.963471)]
#plt.errorbar(window.bjd, window.mag, yerr=window.mag_err, fmt='o', markersize=2, alpha=0.5)
plt.errorbar(fp_f146_df.bjd, fp_f146_df.mag, yerr=fp_f146_df.mag_err, fmt='o', markersize=2, alpha=0.5)
plt.xlabel('BJD')
plt.ylabel('F146 magnitude')
plt.gca().invert_yaxis()
plt.vlines(2.462184e6, 19.85, 20.02, color='r', linestyle='--', label='t0 anomaly')
plt.vlines(12620.373692 + 2450000, 19.85, 20.02, color='g', linestyle='--', label='t0')
plt.legend()
plt.tight_layout()
plt.show()
quit()

"""
for i in tqdm(range(len(fl1_files))):
    fits_image_filename = fl1_files[i] 

    lc_data, names, f146_df = process_event(fl1_path, fits_image_filename, names)

    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)

df_fl1 = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(df_fl1)
df_fl1.to_csv(path+'data/reliability_wg6_fl1.csv', index=False)
"""

"""
for i in tqdm(range(len(ecl_files))):
    # on the 130th iteration, there's something wrong with the 213 wavelength file 
    fits_image_filename = ecl_files[i] 

    try:
        lc_data, names, f146_df = process_event(ecl_path, fits_image_filename, names)
    except Exception as e:
        print(f"Error occurred while processing {fits_image_filename}: {e}") # one such error was that for some reason, a file may not have all 3 filters
        labels.append(np.nan)
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        t0_anomalies.append(np.nan)
        jacscanomaly_scores.append(np.nan)
        continue

    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)

df_ecl = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
df_ecl.to_csv(path+'data/reliability_wg6_ecl.csv', index=False)

for i in tqdm(range(len(t2cep_files))):
    fits_image_filename = t2cep_files[i] 

    try:
        lc_data, names, f146_df = process_event(t2cep_path, fits_image_filename, names)
    except Exception as e:
        print(f"Error processing {fits_image_filename}: {e}")
        labels.append(np.nan)
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        t0_anomalies.append(np.nan)
        jacscanomaly_scores.append(np.nan)
        continue

    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)
df_t2cep = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(df_t2cep)
df_t2cep.to_csv(path+'data/reliability_wg6_t2cep.csv', index=False)
df = pd.concat([df_ecl, df_t2cep], ignore_index=True)
print(df)
df.to_csv(path+'data/reliability_wg6.csv', index=False)
quit()
"""

"""
for i in tqdm(range(len(fl2_files))):
    fits_image_filename = fl2_files[i] 

    lc_data, names, f146_df = process_event(fl2_path, fits_image_filename, names)
    
    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)
df_fl2 = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(df_fl2)
df_fl2.to_csv(path+'data/reliability_wg6_fl2.csv', index=False)
"""

for i in tqdm(range(len(cep_files))):
    fits_image_filename = cep_files[i] 
    try:
        lc_data, names, f146_df = process_event(cep_path, fits_image_filename, names)
    except Exception as e:
        print(f"Error processing {fits_image_filename}: {e}")
        labels.append(np.nan)
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        t0_anomalies.append(np.nan)
        jacscanomaly_scores.append(np.nan)
        continue

    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)
df_cep = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(df_cep)
df_cep.to_csv(path+'data/reliability_wg6_cep.csv', index=False)

for i in tqdm(range(len(dsct_files))):
    fits_image_filename = dsct_files[i] 

    try:
        lc_data, names, f146_df = process_event(dsct_path, fits_image_filename, names)
    except Exception as e:
        print(f"Error processing {fits_image_filename}: {e}")
        labels.append(np.nan)
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        t0_anomalies.append(np.nan)
        jacscanomaly_scores.append(np.nan)
        continue

    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)
df_dsct = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(df_dsct)
df_dsct.to_csv(path+'data/reliability_wg6_dsct.csv', index=False)

for i in tqdm(range(len(ell_files))):
    fits_image_filename = ell_files[i] 

    try:
        lc_data, names, f146_df = process_event(ell_path, fits_image_filename, names)
    except Exception as e:
        print(f"Error processing {fits_image_filename}: {e}")
        labels.append(np.nan)
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        t0_anomalies.append(np.nan)
        jacscanomaly_scores.append(np.nan)
        continue

    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)
df_ell = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(df_ell)
df_ell.to_csv(path+'data/reliability_wg6_ell.csv', index=False)

for i in tqdm(range(len(hb_files))):
    fits_image_filename = hb_files[i] 

    try:
        lc_data, names, f146_df = process_event(hb_path, fits_image_filename, names)
    except Exception as e:
        print(f"Error processing {fits_image_filename}: {e}")
        labels.append(np.nan)
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        t0_anomalies.append(np.nan)
        jacscanomaly_scores.append(np.nan)
        continue
    
    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)
df_hb = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(df_hb)
df_hb.to_csv(path+'data/reliability_wg6_hb.csv', index=False)

for i in tqdm(range(len(lpv_files))):
    fits_image_filename = lpv_files[i] 

    try:
        lc_data, names, f146_df = process_event(lpv_path, fits_image_filename, names)
    except Exception as e:
        print(f"Error processing {fits_image_filename}: {e}")
        labels.append(np.nan)
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        t0_anomalies.append(np.nan)
        jacscanomaly_scores.append(np.nan)
        continue
    
    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)
df_lpv = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(df_lpv)
df_lpv.to_csv(path+'data/reliability_wg6_lpv.csv', index=False)

for i in tqdm(range(len(rrlyr_files))):
    fits_image_filename = rrlyr_files[i] 

    try:
        lc_data, names, f146_df = process_event(rrlyr_path, fits_image_filename, names)
    except Exception as e:
        print(f"Error processing {fits_image_filename}: {e}")
        labels.append(np.nan)
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        t0_anomalies.append(np.nan)
        jacscanomaly_scores.append(np.nan)
        continue
    
    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)
df_rrlyr = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(df_rrlyr)
df_rrlyr.to_csv(path+'data/reliability_wg6_rrlyr.csv', index=False)

for i in tqdm(range(len(t2cep_files))):
    fits_image_filename = t2cep_files[i] 

    try:
        lc_data, names, f146_df = process_event(t2cep_path, fits_image_filename, names)
    except Exception as e:
        print(f"Error processing {fits_image_filename}: {e}")
        labels.append(np.nan)
        t0s.append(np.nan)
        tEs.append(np.nan)
        u0s.append(np.nan)
        t0_anomalies.append(np.nan)
        jacscanomaly_scores.append(np.nan)
        continue

    labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores = detection(lc_data, f146_df, labels, t0s, tEs, u0s, t0_anomalies, jacscanomaly_scores)
df_t2cep = pd.DataFrame({'name': names, 'label': labels, 't0': t0s, 'tE': tEs, 'u0': u0s, 't0_anomaly': t0_anomalies, 'jacscanomaly_score': jacscanomaly_scores})
print(df_t2cep)
df_t2cep.to_csv(path+'data/reliability_wg6_t2cep.csv', index=False)
quit()