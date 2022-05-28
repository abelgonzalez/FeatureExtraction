# Source: https://neurokit2.readthedocs.io/en/latest/examples/ecg_delineate.html
#        https://github.com/neuropsychology/NeuroKit
# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Fig 1
# Visualize R-peaks in ECG signal
def visualize_r_peak(ecg_signal, rpeaks):
    plot = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
    plot.show()


# Fig 2
# Delineate the ECG signal
def delineate_ecg(ecg_signal, rpeaks, sampling_rate):
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate)
    # Visualize the T-peaks, P-peaks, Q-peaks and S-peaks
    plot = nk.events_plot([waves_peak['ECG_T_Peaks'],
                           waves_peak['ECG_P_Peaks'],
                           waves_peak['ECG_Q_Peaks'],
                           waves_peak['ECG_S_Peaks']], ecg_signal)
    plot.show()


# Fig 3
# Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
def zoom_three_rpeaks(waves_peak, ecg_signal, limit):
    plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:3],
                           waves_peak['ECG_P_Peaks'][:3],
                           waves_peak['ECG_Q_Peaks'][:3],
                           waves_peak['ECG_S_Peaks'][:3]], ecg_signal[:limit])
    plot.show()


# Fig 4
# Delineate the ECG signal and visualizing all peaks of ECG complexes
def show_ecg_complexes(ecg_signal, rpeaks, sampling_rate):
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate, show=True, show_type='peaks')

    return _, waves_peak


# Fig 5
# Delineate the ECG signal
def delineate_ecg_signal(ecg_signal, rpeaks, sampling_rate):
    signal_cwt, waves_cwt = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate, method="cwt", show=True,
                                             show_type='all')
    return signal_cwt, waves_cwt


# Main execution:

# Config
plt.rcParams['figure.figsize'] = [8, 5]  # Bigger images
dataset = "ecg_3000hz.csv"
data_column = 'ECG'
sampling_rate = 3000

# Pre-processing
# Retrieve ECG data from data folder (sampling rate= 1000 Hz)
# ecg_signal = nk.data(dataset=dataset)[data_column]
# Extract R-peaks locations and ecg cleaned signal
# _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)

# _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate, show=True, show_type='peaks')

# Visualizations
# Fig 1
# Visualize R-peaks in ECG signal
# visualize_r_peak(ecg_signal, rpeaks)

# Fig 2
# Delineate the ECG signal
# delineate_ecg(ecg_signal, rpeaks, sampling_rate)

# Fig 3
# Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
limit = 12500
# zoom_three_rpeaks(waves_peak, ecg_signal, limit)

# Fig 4
# Delineate the ECG signal and visualizing all peaks of ECG complexes
# show_ecg_complexes(ecg_signal, rpeaks, sampling_rate)


# Fig 5
# Delineate the ECG signal
# delineate_ecg_signal(ecg_signal, rpeaks, sampling_rate)

# Other informations
# df2 = pd.DataFrame(waves_peak)
# list_Peaks = next(iter(waves_peak.values()))
# print(df2)

# print("New cicle (P_Peaks) starts in: " + str(list_Peaks))

# Visualizing an resumed information from dataset.
WFDB_FILE = os.getcwd() + '\\data\\Test_ecg7.csv'
data2 = pd.read_csv(WFDB_FILE)
data_column2 = 'ECG0'

# Normalization =(B2-MIN($B$2:$B$2741))/(MAX($B$2:$B$2741)-MIN($B$2:$B$2734))

# ------> Global variables
FREQUENCY = 100  # Hz
DURATION = 12  # seconds
DATAPOINTS = DURATION * FREQUENCY

sampling_rate2 = DATAPOINTS

# sampling_rate (3000Hz) in which the signals were generated
ecg_signals2, info = nk.ecg_process(data2[data_column2], sampling_rate=DATAPOINTS)
print(nk.ecg_intervalrelated(ecg_signals2))

# plt = nk.ecg_plot(ecg_signal[:1000], sampling_rate=3000)  # 3000 datapoints (or 10s) to visualize.


ecg_signal = data2[data_column2]
# Extract R-peaks locations and ecg cleaned signal
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate2)
_, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate2, show=True, show_type='peaks')

df2 = pd.DataFrame(waves_peak)
list_Peaks = next(iter(waves_peak.values()))
print(df2)

print("New cicle (P_Peaks) starts in: " + str(list_Peaks))

df2.to_csv(os.getcwd() + '/exported/Mergulhadores.csv', na_rep='NaN')

df3 = pd.DataFrame(waves_peak)
list_Peaks3 = next(iter(waves_peak.values()))
print("Events (R_Peaks) occurs in: " + str(list_Peaks3))

visualize_r_peak(ecg_signal, rpeaks)

plt = nk.ecg_plot(ecg_signals2, sampling_rate=FREQUENCY)
plt.show()
