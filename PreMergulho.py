# Source: https://neurokit2.readthedocs.io/en/latest/examples/ecg_delineate.html
#        https://github.com/neuropsychology/NeuroKit
# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
plt.rcParams['font.size'] = 13


# --------> Normalization formula (Excel)
# =(B2-MIN($B$2:$B$2741))/(MAX($B$2:$B$2741)-MIN($B$2:$B$2734))


# --------> Functions
# Visualize R-peaks in ECG signal
def visualize_r_peak(ecg_signal, rpeaks):
    plot = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
    plot.show()


# ----> Global variables

# --> Read file and column name
# WFDB_FILE = os.getcwd() + '\\data\\MIT_ecg.csv'
# WFDB_FILE = os.getcwd() + '\\data\\Test_ecg11.csv'
WFDB_FILE = os.getcwd() + '\\data\\BioPac.csv'
df = pd.read_csv(WFDB_FILE)
ECG_COLUMN_NAME = 'Ch1'

# --> File settings
# Input sample: MIT_ecg.csv it contains 10 minutes of physiological signals recorded
# at a frequency of 360Hz (10 x 60 x 360 = 216000 data points).
# SAMPLING_RATE = 1000  # Ex: 360 Hz (samples/second) in which the signals were generated
SAMPLING_RATE = 125  # 4 msec/sample , aprox 124.89 Hz
# DURATION = 10  # Ex:  600 seconds (10 minutes)
DURATION = 3600  # Ex:  3600 seconds (60 minutes)
DATAPOINTS = DURATION * SAMPLING_RATE  # Ex: 216000 data points. It is not necessary to be exact.

# 1 hour = 60 min = 3600 sec

# Setting ECG interval to be processed:
ecg_interval = df[ECG_COLUMN_NAME]
ecg_interval = ecg_interval[:6250]  # 6250 datapoints (or 5s) to visualize

# --------> Main execution:
# Setting "ecg_signals" variable (yes, it in plural) for the full visualization
ecg_signals, info = nk.ecg_process(ecg_interval, sampling_rate=SAMPLING_RATE)
plt = nk.ecg_plot(ecg_signals, sampling_rate=SAMPLING_RATE)
plt.show()

# Extract R-peaks locations and ecg cleaned signal
_, rpeaks = nk.ecg_peaks(ecg_interval, sampling_rate=SAMPLING_RATE)
_, waves_peak = nk.ecg_delineate(ecg_interval, rpeaks, sampling_rate=SAMPLING_RATE, show=True, show_type='peaks')

visualize_r_peak(ecg_interval, rpeaks)

df2 = pd.DataFrame(waves_peak)
list_Peaks = next(iter(waves_peak.values()))
print(df2)

print("New cicle (P_Peaks) starts in: " + str(list_Peaks))

df2.to_csv(os.getcwd() + '/exported/Pre_Mergulho-ECG.csv', na_rep='NaN')

df3 = pd.DataFrame(waves_peak)
list_Peaks3 = next(iter(waves_peak.values()))
print("Events (R_Peaks) occurs in: " + str(list_Peaks3))
