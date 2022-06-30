import os
import wfdb
import wfdb.processing
import numpy as np
from sklearn.model_selection import train_test_split
#from torch import nn, optim
#import torch
import copy
import seaborn as sns
import os
import scipy
from scipy.signal import resample, medfilt
from mvgavg import mvgavg

import ipywidgets as widgets

normalize_it = True
Lower_Bound = 0
Upper_Bound = 1

SKIP = 5
num_records = 216000
filter_enable_butterworth = True
filter_bw_type = 'highpass'
filter_bw_cutoff = 0.5

filter_enable_moving = True
filter_moving_type = 'median'
filter_moving_window_size = 5
filter_hampel_sigmas = 3

# matplotlib inline
import matplotlib.pyplot as plt

plt.style.use('classic')

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import wfdb

WFDB_FILES_DIR = os.getcwd() + '/datasetMIT/'


def get_records(WFDB_FILES_DIR):
    """
      Get from input the list separated by commas with the number of the
      records to be downloaded or read.

      It checks in WFDB_FILES_DIR if the files are already there. If not,
      it downloads the required files according with the list of selected
      records.

      Parameters
      ----------
      WFDB_FILES_DIR : Path from where WFDB files should be read/stored

      Returns
      -------
      l_files : List of read files (with extensions .dat, .atr and .hea)
      l_subjects : List of read subjects
    """
    print("Enter with the list of records, i.e.:  112, 200")

    line = ""
    while True:
        line = input("Enter:")
        recs = line.split(",")
        if len(line) > 0:
            break

    l_files = []
    l_subjects = []
    for rec in recs:
        rec = rec.strip()
        l_subjects.append(rec)

        if not os.path.exists(WFDB_FILES_DIR + rec + ".dat"):
            files = [rec + ".dat", rec + ".atr", rec + ".hea"]
            l_files.extend(files)

    # Download missing files
    if len(l_files) > 0:
        # Database identification, where to store, list of files to be downloaded
        wfdb.dl_files('mitdb', WFDB_FILES_DIR, files=l_files)

    return l_files, l_subjects


l_files, l_subjects = get_records(WFDB_FILES_DIR)

print("\n Files ready to be used.")


def plot_waves(waves, step):
    """
    Plot a set of 9 waves from the given set, starting from the first one
    and increasing in index by 'step' for each subsequent graph

    Parameters
    ----------
    waves : 1D array or list with the segments
    step  : Integer. Steps between chosen segments to be plotted

    Returns
    -------
    None.
    """

    plt.figure()
    n_graph_rows = 3
    n_graph_cols = 3
    graph_n = 1
    wave_n = 0
    for _ in range(n_graph_rows):
        for _ in range(n_graph_cols):
            axes = plt.subplot(n_graph_rows, n_graph_cols, graph_n)
            axes.set_ylim([-2, 2.0])
            plt.plot(waves[wave_n])
            graph_n += 1
            wave_n += step
    # fix subplot sizes so that everything fits
    plt.tight_layout()
    plt.show()


def filter_butterworth(in_signal, order, fs, fc, btype='low'):
    """
    Apply butterworth filter.

    Parameters
    ----------
    in_file : 1D array
        Original signal to be filtered
    order : Integer
        Filter magnitude order
    fs : Float
        Captured frequency.
    fc : Float
        Cut-off frequency
    btype : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        The type of filter. Default is ‘lowpass’.

    Returns
    -------
    Filtered signal
    """
    nyq_freq = fc / (fs / 2)
    b, a = scipy.signal.butter(order, nyq_freq, btype)
    output = scipy.signal.filtfilt(b, a, in_signal)

    end = 150
    if len(in_signal) < 150:
        end = len(in_signal)

    # Look at the examples
    plt.title("Butterworth filter order: " + str(order) + " cut-off: " + str(fc))
    line1, = plt.plot(in_signal[:end], label='raw')
    line2, = plt.plot(output[:end], label='filtered')
    plt.legend(handles=[line1, line2], loc='upper right')
    plt.minorticks_on()
    plt.show()

    return output


def filter_moving_average(signal, window=10, binning=False):
    """
    Perform moving average filter on signal.

    Parameters
    ----------
    signal : 1D array
        Time series to be filtered
    window : Integer
        Size of moving window, optional
        The default is 10.
    binning : Boolean, optional
        Separate windows or sliding windows. The default is False.

    Returns
    -------
    TYPE
        Filtered signal.

    """
    return mvgavg(signal, window, binning)


from scipy.signal import medfilt


def filter_moving_median(signal, window_size=7):
    """
    Filter the time series using moving median (instead of moving average)

    Parameters
    ----------
    signal : 1D array
        Time series
    window_size : Integer, optional
        Size of the windows. It must be odd. The default is 9.

    Returns
    -------
    TYPE
        The filtered signal
    """

    if window_size % 2 == 0:
        print("Windows size must be odd")
        return None
    return medfilt(signal, np.array(window_size))


def filter_hampel(signal, window_size=7, n_sigmas=3):
    """
    Filter the signal using Hampel Filter. It is a moving median filtered
    with a threshold.

    Parameters
    ----------
      signal : 1D array
          Time series.
      window_size : Integer, optional
          Size of the window. The default is 7.
      n_sigmas : Float, optional
          Outlier threshold. Number of standard deviations far from median.
          The default is 3.

    Returns
    -------
      new_series : 1D array
          Filtered signal.

    """
    n = len(signal)

    new_series = signal.copy()
    k = 1.4826  # scale factor for Gaussian distribution

    # possibly use np.nanmedian
    for i in range((window_size), (n - window_size)):
        x0 = np.median(signal[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(signal[(i - window_size):(i + window_size)] - x0))
        if (np.abs(signal[i] - x0) > n_sigmas * S0):
            new_series[i] = x0

    return new_series


#
# Cut off initial and end parts of beats that have length greather than size
#
def sameSizeBeatsFill(beats, size, fill=False):
    newSerie = []

    for beat in beats:
        beat_size = len(beat)

        if beat_size == size:
            newSerie.append(beat)
        elif beat_size > size:
            diff = int(beat_size - size)

            start = int(diff // 2)
            end = start

            if len(beat[:-diff]) <= 0:
                print(diff, start, end)

            newSerie.append(beat[:-diff])
        elif fill:
            diff = int(size - beat_size) + 1
            beat.extend(np.zeros(diff))
            newSerie.append(beat)
    return newSerie


def same_size_beats(beats, seg_size):
    """
    Gets variable heart beat duration, it means, variable number of
    samples per heart beat and resample them to a fixed heart beat size.

    Parameters
    ----------
    signal : 2D array
            List of heart beats. Each heart beat is a time series.

    seg_size : integer
        The size of the desired segment.

    Returns
    -------
    A list with every heart beat with the same size.
    """

    norm_size_beats = []

    for beat in beats:
        new_beat = resample(beat, seg_size)
        norm_size_beats.append(new_beat)

    return norm_size_beats


def same_size_padding_beats(beats, size, padding_value=0):
    """
    Receives a list of heart beats with variable sizes.

    Smaller beats are filled up with padding_value up to size.
    Larger beats are resampled to size.
    Same size beats remain the equal.

    Parameters
    ----------
    beats : List or 1D array
        List with beat segments
    size : Integer
        Desired size

    padding_value : Integer, optional
        Padding value. The default is 0.

    Returns
    -------
    List with beats with length >= size.
    """
    newSerie = []

    for beat in beats:
        beat_size = len(beat)

        if beat_size > size:
            beat = resample(beat, size)
        elif beat_size < size:
            diff = int(size - beat_size)
            beat = np.hstack([beat, np.zeros(diff)])

        newSerie.append(beat)

    return newSerie


def split_normal_abnormal_beats(signal, samples, symbols):
    """
    This is specific for MIT datasets - WFDB

    Gets the raw signal and the list with the location of each beat (samples).
    It also receives the label (classification) of each beat.

    Split the raw signal in Normal Beats (label == 'N') and Abnormal Beats
    (label != 'N').

    It also calculates the mean beat size.


    Parameters
    ----------
    signal :    1D array
                Raw ECG time series - 1 channel
    samples :   1D array
                List with the position of each heart beat.
                Its size is the number of identified heart beats in the signals
    symbols :   1D array
                It has for each heartbeat its label. 'N' means normal.

    Returns
    -------
    List with all Beats
    List with Normal Beats
    List with Abnormal Beats (all beats different of 'N')
    Mean beat size.
    """
    i = 0
    averageSize = 0

    normalBeats = []
    abnormalBeats = []
    allBeats = []
    totalBeats = len(samples)

    # Get each beat start and end positions, and the correspondent label
    for start, symbol in zip(samples, symbols):
        if i < (totalBeats - 1):
            averageSize += samples[i + 1] - samples[i]
            end = samples[i + 1]
        else:
            end = len(signal)

        beat = signal[start:end]

        if symbol == 'N':
            normalBeats.append(beat)
        else:
            abnormalBeats.append(beat)

        allBeats.append(beat)
        i += 1

    averageSize = averageSize // totalBeats

    return allBeats, normalBeats, abnormalBeats, averageSize


def get_float_number(lbl_input, lbl_feedback, default_value):
    """
      Request a float number.

      Parameters
      ----------
      lbl_input : String
          Text for requesting the number.

      lbl_feedback : String
          Text showing the chosen value
      default_value : Float positive number
          Default value

      Returns
      -------
      val : Float
          Default value or inputed float valued.

      """
    val = 0

    while val == 0:
        val = input(lbl_input)

        if not val:
            val = default_value
        else:
            try:
                float(val)
                val = float(val)

                if val <= 0:
                    print("Must be a positive number")
                    val = 0
            except ValueError:
                print("Must be a positive number")
                val = 0

    print(lbl_feedback, val)
    return val


"""
import wfdb.processing
wfdb.processing.resample_sig(x, fs, fs_target)
  Resample a signal to a different frequency.

  x : ndarray
      Array containing the signal.

  fs : int, float
      The original sampling frequency.

  fs_target : int, float
    The target frequency.

  resampled_x : ndarray
      Array of the resampled signal values.

  resampled_t : ndarray
      Array of the resampled signal locations.
"""

print("List of records to be loaded and used for training, validating and testing the model.")

# Records to be read. Description of each record
# https://physionet.org/physiobank/database/html/mitdbdir/records.htm
# Subjects only males: (id, Age <= 65)
#  (103, ni), (109,64), (112,54), (122,51), (200,64), (203,43)
#  (205, 59), (209, 62), (213, 61), (214,53), (217, 65), (219, ni) (230, 32), (233, 57)

ans = ""
while ans.lower() != "y":
    print(l_subjects)
    ans = input("Confirm [Y/n]?")

    if ans.lower() == "n":
        l_files, l_subjects = get_records()

# bw_cutoff = get_float_number("Enter butterworh highpass cutoff [0.5]:",  "Chosen Butterworth Highpass cut-off frequency:", 0.5)

# Get record frequency
hea = wfdb.rdheader(WFDB_FILES_DIR + l_subjects[0])
FS = hea.fs

#####################
# Global variables
#####################
PRINT_MSGS = True
PLOT_GRAPHS = True

# Number of samples
N = (SKIP) + num_records

# Each record has the raw signal and the header
l_signals = []
l_headers = []
# Correspondent Annotation file has signal labels
l_labels = []

#######################################################################
#                       Load selected files
#######################################################################
for subject in l_subjects:
    # read subject signal
    record, header = wfdb.rdsamp(WFDB_FILES_DIR + subject, sampfrom=SKIP, sampto=N, channels=[0])

    fs = header['fs']
    # record = wfdb.rdrecord(baseDir + subject, sampfrom=SKIP, sampto=N, channels=[0], physical=False)
    rec = [x[0] for x in record]

    if filter_enable_butterworth:
        # Pass a butterworth highpass filter for handling baseline wander
        rec = filter_butterworth(rec, 3, fs, filter_bw_cutoff, filter_bw_type)

    if filter_enable_moving:
        if filter_moving_type == "Median":
            # Pass a moving median filter on signal
            rec = filter_moving_median(rec, window_size=3)
        elif filter_moving_type == "Average":
            rec = filter_moving_average(rec, window=filter_moving_window_size)
        elif filter_moving_type == "Hampel":
            rec = filter_hampel(rec, windows_size=filter_moving_window_size, n_sigmas=filter_hampel_sigmas)

    if normalize_it:
        rec = wfdb.processing.normalize_bound(rec, lb=Lower_Bound, ub=Upper_Bound)

    l_signals.append(rec)
    l_headers.append(header)

    # read subject annotations
    ann = wfdb.rdann(WFDB_FILES_DIR + subject, 'atr', sampfrom=SKIP, sampto=N)

    l_labels.append(ann.__dict__)

#######################################################################
#                Find and Filter Anomalies
#        Produce: normalSignal and abnormalSignal (not used)
#        Calculate mean beat size (mean number of samples per beat)
#                  it is equivalent to R-R interval
#######################################################################
l_average_size = []  # average beat size list
normalSignal = []  # signal without anomalies
normalBeat = []  # list with all normal beats
abnormalBeat = []  # list with all abnormal beats
allBeats = []
allSymbols = []

num_beats = 0  # Total number of beats
labelled_anomalies = 0  # Total number of anomalies

for signal, header, label in zip(l_signals, l_headers, l_labels):
    # it has the position of each beat. The position should be adjusted
    # if we skip some beats
    samples = label['sample'] - SKIP
    symbols = label['symbol']

    num_beats += len(samples)
    fs = header['fs']

    labelled_anomalies += (len(symbols) - symbols.count('N'))

    allBeat, normal, abnormal, avg = split_normal_abnormal_beats(signal, samples, symbols)
    l_average_size.append(avg)

    allBeats.extend(allBeat)
    normalBeat.extend(normal)
    abnormalBeat.extend(abnormal)
    allSymbols.extend(symbols)

    mean_rr_interval = (1 / fs) * avg
    print("Individual Mean R-R interval %.2f sec - FC ~= %d bpm - Average size %d #beats %d" %
          (mean_rr_interval, 60 / mean_rr_interval, avg, len(samples)))

# Average Segment length
l_average_size = np.array(l_average_size)
beat_size = int(l_average_size.mean())

if PRINT_MSGS:
    print("Normal signal")
    print("Number of beats %d \t Mean beat size %d" % (num_beats - len(abnormalBeat), l_average_size.mean()))

    mean_rr_interval = (1 / fs) * l_average_size.mean()
    if PRINT_MSGS:
        print("Total Mean R-R interval %.2f sec - FC ~= %d bpm" % (mean_rr_interval, 60 / mean_rr_interval))

    print("\nOriginal signal")
    print("Number of beats  %d - anomalies %d" % (num_beats, labelled_anomalies))

    print(f"You can choose a new beat size. The default is the average heart beat size of loaded records: {beat_size}")

    wbeat_size = widgets.IntText(
        value=beat_size,
        description='Beat Size:',
        disabled=False
    )

    #display(wbeat_size)


# Saving MIT-BIH first list record to CSV file

import pandas as pd

ecg_signal = pd.DataFrame(list(zip(*[l_signals[0]]))).add_prefix('ECG')
ecg_signal.to_csv(os.getcwd() + '\\data\MIT_ecg.csv')

ecg_signal_rPeaks = pd.DataFrame(list(zip(*[samples]))).add_prefix('R_Peaks')
ecg_signal_rPeaks.to_csv(os.getcwd() + '\\data\MIT_ecg_RPeaks.csv')

print(ecg_signal)
