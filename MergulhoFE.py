import os

import neurokit2 as nk
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing

# Show number of rows and columns when all rows are displayed
pd.set_option('show_dimensions', True)

# It contains 5 minutes of physiological signals recorded at a frequency of 100Hz (5 x 60 x 100 = 30000 data points).
# Dataset --------> "bio_resting_5min_100hz"
# Sampling rate --> 100 Hz (samples/second)
# Data points --------> 30000 (start with 0)
# Total Samples --> 430 cardiac cycles
# Duration -------> 5 minutes / 300 seconds
# Mean Beat amplitude -> xx
# ------> Global variables
# FREQUENCY = 100  # Hz
# DURATION = 300  # seconds
# DATAPOINTS = DURATION * FREQUENCY
# DATASET = os.getcwd() + '/data/bio_resting_5min_100hz.csv'
# ECG_COLUMN_NAME = 'ECG'

# Dataset --------> "MIT_ecg.csv"
# Sampling rate --> 360 Hz (samples/second)
# Data points --------> 216000 (start with 0)
# Total Samples --> 855 cardiac cycles (1 anomaly)
# Duration -------> 10 minutes / 600 seconds
# Mean Beat amplitude -> 252 (mean)
# ------> Global variables
FREQUENCY = 125  # Hz
DURATION = 3600  # seconds
DATAPOINTS = DURATION * FREQUENCY
DATASET = os.getcwd() + '/data/BioPac.csv'
# DATASET = os.getcwd() + '/data/Test_ecg.csv'
ECG_COLUMN_NAME = 'Ch1'


# ------> Functions
# Function to calculate data peaks interval (miliseconds)

def interval_miliseconds(time1, time2):
    interval_time = time2 - time1
    # DURATION = DATAPOINTS / FREQUENCY
    interval_time_milis = interval_time * 1000 / FREQUENCY
    return interval_time_milis


def interval(data_peaks_df, ecg_initial, ecg_final):
    """
        Calculates data peaks interval.

        Parameters
        ----------
        data_peaks : DataFrame Pandas
        ecg_initial : String. Ex 'ECG_Q_Peaks'
        ecg_final : String. Ex 'ECG_T_Peaks'

        Returns
        -------
        dataFrame : Pandas dataframe with data peaks interval requested.

      """

    # data_frame = pd.concat([data_frame_aux, y_values_current_segment], axis=1)

    data_frame = data_peaks_df[[ecg_initial, ecg_final]]
    # data_frame = data_peaks_df

    # interval_time = interval_miliseconds(data_frame[ecg_initial], data_frame[ecg_final])
    interval_time = data_frame[ecg_final] - data_frame[ecg_initial]
    # DURATION = DATAPOINTS / FREQUENCY
    # interval_time_milis = interval_time * 1000 / FREQUENCY

    # Delete all columns except the last one
    # columns_quant = len(dataframe.columns) - 1
    # dataframe.drop(dataframe.iloc[:, 1:columns_quant], inplace=True, axis=1)

    initial_char = ecg_initial  # [4]
    final_char = ecg_final  # [4]
    data_frame.insert(2, initial_char + '->' + final_char + ' ' + 'interval(miliseconds)', interval_time)
    data_frame.index.name = 'Cardiac Cycle (beat)'
    # data_frame.index += 1

    return data_frame


def interval_2(data_frame, ecg_initial_str, ecg_final_str):
    """
        Calculates RR2 interval. Means the distance between the x value of R of the current
        segment and x value of R-peak of the next-but-one segment.

        Parameters
        ----------
        data_frame : DataFrame Pandas
        ecg_initial_str : String. Ex 'Rx'
        ecg_final_str : String. Ex 'ECG_R_Peaks'

        Returns
        -------
        dataFrame : Pandas dataframe with data peaks interval requested.

      """

    # Create an duplicate 'ECG_Peaks' column
    data_frame_aux = pd.DataFrame(data_frame[ecg_final_str])

    # Copy a new column with same values
    data_frame_aux[ecg_final_str] = data_frame[ecg_final_str]

    # Select the new column, delete the first item, fill with 0 the last position
    col_aux_list = data_frame[ecg_final_str].tolist()
    col_aux_list.pop(0)
    col_aux_list.insert(len(col_aux_list), 0)
    # The new column will be the modified list
    data_frame_aux[ecg_final_str] = col_aux_list

    # Insert the 'x' columns
    data_frame_aux[ecg_initial_str] = data_frame[ecg_initial_str]

    # Calculate the diference
    interval_df = interval(data_frame_aux, ecg_initial_str, ecg_final_str)

    initial_char = ecg_initial_str[0]
    str_interval = initial_char + initial_char + '2'

    data_frame_aux = interval_df
    # Delete columns one and two, rename the third
    data_frame_aux.drop(ecg_initial_str, axis=1, inplace=True)
    data_frame_aux.drop(ecg_final_str, axis=1, inplace=True)
    data_frame_aux.rename(columns={data_frame_aux.columns[0]: str_interval}, inplace=True)

    return data_frame_aux


def interval_same(data_frame, ecg_signal_str):
    """
            Calculates data peaks interval for the same peak.

            Parameters
            ----------
            data_peaks : DataFrame Pandas
            ecg_signal : String. Ex 'ECG_R_Peaks'

            Returns
            -------
            dataFrame : Pandas dataframe with data peaks interval requested (R-->R or T-->T).

          """

    # peak_char = ecg_signal[4]

    # Copy a new column with same values
    data_frame_aux = pd.DataFrame(data_frame[ecg_signal_str])

    data_frame_aux[ecg_signal_str] = data_frame[ecg_signal_str]
    # Select the new column, delete the first item, fill with 0 the last position
    col_aux_list = data_frame[ecg_signal_str].tolist()
    col_aux_list.pop(0)
    col_aux_list.insert(len(col_aux_list), 0)
    # The new column will be the modified list
    data_frame_aux[ecg_signal_str] = col_aux_list

    interval_df = (data_frame_aux[ecg_signal_str] - data_frame[ecg_signal_str])

    if len(ecg_signal_str) == 2:
        initial_char = ecg_signal_str[0]
    else:
        initial_char = ecg_signal_str[4]

    str_interval = initial_char + initial_char + 'interval'

    data_frame_aux[str_interval] = interval_df

    data_frame_aux.drop(ecg_signal_str, axis=1, inplace=True)

    # data_frame_aux['Time (miliseconds)'] = interval_miliseconds(data_frame[ecg_signal_str],
    #                                                            data_frame[ecg_signal_str])

    return data_frame_aux


def average(data_peaks, ecg_initial, ecg_final):
    """
        Calculates data peaks average.

        Parameters
        ----------
        data_peaks : DataFrame Pandas
        ecg_initial : String. Ex 'ECG_P_Peaks'
        ecg_final : String. Ex 'ECG_Q_Peaks'

        Returns
        -------
        dataFrame : Pandas dataframe with average data peaks interval requested.

      """

    data_frame = data_peaks[[ecg_initial, ecg_final]]

    # interval_time = interval_miliseconds(data_frame[ecg_initial], data_frame[ecg_final])
    interval_time = (data_frame[ecg_final] + data_frame[ecg_initial]) / 2
    # DURATION = DATAPOINTS / FREQUENCY
    # interval_time_milis = interval_time * 1000 / FREQUENCY

    if len(ecg_initial) == 2 or len(ecg_final) == 2:
        initial_char = ecg_initial
        final_char = ecg_final
    else:
        initial_char = ecg_initial[4]
        final_char = ecg_final[4]

    data_frame.insert(2, initial_char + '->' + final_char + ' ' + 'Average(miliseconds)', interval_time)
    data_frame.index.name = 'Cardiac Cycle (beat)'
    # data_frame.index += 1

    return data_frame


def normalize_column(dataframe):
    """
               Normalizing the elements of a Pandas DataFrame column converts each element in a DataFrame
               to a value between 0 and 1.

               Parameters
               ----------
               dataframe : DataFrame Pandas column to be normalized


               Returns
               -------
               dataFrame : Pandas dataframe normalized

             """
    # Mean normalization
    # normalized_df = (dataframe - dataframe.mean()) / dataframe.std()

    # Min-max normalization
    normalized_df = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

    return normalized_df


def extract_feature(dataframe, feature):
    """
                Extract features from dataframe

                Parameters
                ----------
                dataframe : DataFrame Pandas
                feature : String. Ex 'ECG_Q_Peaks'

                Returns
                -------
                dataFrame : Pandas dataframe feature

              """
    # Dummy variable result for missing data
    result = pd.DataFrame([np.nan] * dataframe.__len__())

    # Rename column
    result.columns = [feature]

    if feature in dataframe.columns:
        result = dataframe[feature]

    return result


def extract_x(data_peaks, ecg_initial, ecg_final):
    """
                Extract features from dataframe
                The x values in each segment are defined as
                the relative distance to the location of P-wave as a reference point.


            Parameters
            ----------
            data_peaks : DataFrame Pandas
            ecg_initial : String. Ex 'ECG_P_Peaks'
            ecg_final : String. Ex 'ECG_T_Peaks'

            Returns
            -------
            dataFrame : Pandas dataframe column with data peaks interval requested.

          """

    # Case P_Peaks x to P_Peaks x
    if ecg_initial == ecg_final:
        dataframe_aux = [0 for _ in range(len(data_peaks))]
        dataframe = pd.DataFrame(dataframe_aux)
    else:
        dataframe = interval(data_peaks, ecg_initial, ecg_final)
        columns = [ecg_initial, ecg_final]

        # Delete columns ecg_initial, ecg_final and stay with distance between them
        dataframe.drop(columns, inplace=True, axis=1)

        # Rename column
    peak_char = ecg_final[4]
    new_column_name = peak_char + 'x'
    dataframe.columns = [new_column_name]

    return dataframe


def extract_y(ecg_column_signals, ecg_data, ecg_peaks_str):
    """
                Extract features from dataframe
                The x values in each segment are defined as
                the relative distance to the location of P-wave as a reference point.


            Parameters
            ----------
            ecg_column_signals : DataFrame Pandas of ECG_Raw signal
            ecg_data : DataFrame Pandas of ECG Peaks
            ecg_peaks_str : String peaks. Ex 'ECG_P_Peaks'

            Returns
            -------
            result_dataframe : Pandas dataframe column with the amplitude requested.

          """

    ecg_pandas_peaks = ecg_data[ecg_peaks_str]
    # Electrical impulses dataframe
    result_list = []

    for a in ecg_pandas_peaks:
        # In case of a contain decimals (OnQRS_y and OffQRS_y),
        # truncate the value and get the nearly ECG voltage signal.
        a_truncated = math.trunc(a)
        result_list.append(ecg_column_signals.__getitem__(a_truncated))

    # Column name
    peak_char = ecg_peaks_str[4]
    column_name = peak_char + 'y'

    result_dataframe = pd.DataFrame(result_list, columns=[column_name])

    return result_dataframe


def extract_QRS_x(data_peaks, ecg_initial, ecg_final, prefix):
    """
                Extract features from dataframe
                The x values in each segment are defined as
                the relative distance to the location of P-wave as a reference point.


            Parameters
            ----------
            data_peaks : DataFrame Pandas
            ecg_initial : String. Ex 'Px'
            ecg_final : String. Ex 'Qx'

            Returns
            -------
            dataFrame : Pandas dataframe column with data peaks interval requested.

          """

    # Case P_Peaks x to P_Peaks x
    if ecg_initial == ecg_final:
        dataframe_aux = [0 for _ in range(len(data_peaks))]
        dataframe = pd.DataFrame(dataframe_aux)
    else:
        dataframe = average(data_peaks, ecg_initial, ecg_final)
        columns = [ecg_initial, ecg_final]
        # Delete columns ecg_initial, ecg_final and stay with distance between them
        dataframe.drop(columns, inplace=True, axis=1)

        # Rename column
    new_column_name = prefix + 'QRS_x'
    dataframe.columns = [new_column_name]

    return dataframe


def extract_QRS_y(ecg_column_signals, df_merged_col, qrs_x, prefix):
    """
                Extract features from dataframe
                The x values in each segment are defined as
                the relative distance to the location of P-wave as a reference point.


            Parameters
            ----------
            ecg_column_signals : DataFrame Pandas
            df_merged_col : DataFrame Pandas
            qrs_x : String. Ex 'OnQRS_x'
            prefix : String. Ex 'On'

            Returns
            -------
            dataFrame : Pandas dataframe column with data peaks interval requested.

          """

    dataframe_res = extract_y(ecg_column_signals, df_merged_col, qrs_x)

    # Rename column
    new_column_name = prefix + 'QRS_y'
    dataframe_res.columns = [new_column_name]

    return dataframe_res


def amplitude_(df_ecg_signals, df_merged_col, str_ecg):
    """
                    Extract amplitude feature from dataframe
                    R−Ramplitude means the difference between the y value
                    of the R-peak of the current segment and the y value of
                    the R-peak of the next segment.


                Parameters
                ----------
                df_ecg_signals : DataFrame Pandas
                df_merged_col : DataFrame Pandas
                str_ecg : String. Ex 'ECG_P_Peaks'

                Returns
                -------
                dataFrame : Pandas dataframe column with data peaks interval requested.

              """
    y_values_current_segment = extract_y(df_ecg_signals, df_merged_col, str_ecg)

    # Create an duplicate 'ECG_y' column
    data_frame_aux = y_values_current_segment

    # Column name
    peak_char = str_ecg[4]
    column_y_name = peak_char + 'y'

    # result_dataframe = pd.DataFrame(result_list, columns=[column_name])

    # Copy a new column with same values
    data_frame_aux[column_y_name + '1'] = y_values_current_segment  # df_merge_col[str_ecg]

    # Select the new column, delete the first item, fill with 0 the last position
    col_aux_list = data_frame_aux[column_y_name + '1'].tolist()  # df_merge_col[str_ecg].tolist()
    col_aux_list.pop(0)
    col_aux_list.insert(len(col_aux_list), 0)
    # The new column will be the modified list
    data_frame_aux[column_y_name + '2'] = col_aux_list

    # Delete column one
    data_frame_aux.drop(column_y_name, axis=1, inplace=True)

    # Calculate the diference
    interval_df = interval(data_frame_aux, column_y_name + '1', column_y_name + '2')

    data_frame_aux = interval_df

    # Delete columns one and two, rename the third
    data_frame_aux.drop(column_y_name + '1', axis=1, inplace=True)
    data_frame_aux.drop(column_y_name + '2', axis=1, inplace=True)

    column_name = peak_char + '-' + peak_char + 'amplitude'

    data_frame_aux.rename(columns={data_frame_aux.columns[0]: column_name}, inplace=True)

    return data_frame_aux


# data_frame, ecg_initial_str, ecg_final_str):
def amplitude_2(df_ecg_signals, df_merged_column, string_ecg):
    """
        R − R2amplitude means the difference between the y value of R-peak of the current
                              segment and the R-peak of the next-but-one segment.



                Parameters
                ----------
                df_ecg_signals : DataFrame Pandas
                df_merged_column : DataFrame Pandas
                string_ecg : String. Ex 'ECG_P_Peaks'

                Returns
                -------
                dataFrame : Pandas dataframe column with data peaks interval requested.

              """
    # ----> RyCurrent - RPeak_YNext
    # Create an duplicate 'ECG_Peaks' column
    data_frame_aux = pd.DataFrame(df_merged_column[string_ecg])
    # data_frame_aux = extract_y(df_ecg_signals, df_merged_column, string_ecg)
    y_values_current_segment = extract_y(df_ecg_signals, df_merged_column, string_ecg)

    # Column name
    peak_char = string_ecg[4]
    column_peak_name = peak_char + 'y'

    # Copy a new column with same values
    data_frame_aux[string_ecg] = df_merged_column[string_ecg]

    # Select the new column, delete the first item, fill with 0 the last position
    col_aux_list = df_merged_column[string_ecg].tolist()
    col_aux_list.pop(0)
    col_aux_list.insert(len(col_aux_list), 0)
    # The new column will be the modified list
    data_frame_aux[string_ecg] = col_aux_list

    y_values_next_peak_segment = extract_y(df_ecg_signals, data_frame_aux, string_ecg)
    y_values_next_peak_segment.rename(columns={y_values_next_peak_segment.columns[0]: column_peak_name + '2'},
                                      inplace=True)

    # Insert the 'x' columns
    data_frame_aux[column_peak_name] = y_values_current_segment
    data_frame_aux[column_peak_name + '2'] = y_values_next_peak_segment

    # Calculate the diference
    interval_df = interval(data_frame_aux, column_peak_name, column_peak_name + '2')

    data_frame_aux = interval_df

    # Delete columns one and two, rename the third
    data_frame_aux.drop(column_peak_name, axis=1, inplace=True)
    data_frame_aux.drop(column_peak_name + '2', axis=1, inplace=True)

    column_name = peak_char + '-' + peak_char + '2amplitude'

    data_frame_aux.rename(columns={data_frame_aux.columns[0]: column_name}, inplace=True)

    return data_frame_aux


def RR_interval(df_merge_col, string_ecg):
    """
            RR Interval.

                    Parameters
                    ----------
                    df_merge_col : DataFrame Pandas
                    string_ecg : String. Ex 'ECG_P_Peaks'

                    Returns
                    -------
                    dataFrame : Pandas dataframe column with data peaks interval requested.

                  """

    # -RR interval

    lj = extract_feature(df_merge_col, string_ecg)

    # df_lj_34 = pd.concat([lj3, lj4], axis=1)

    lj1 = extract_feature(df_merge_col, string_ecg)
    # lj1 = lj1.drop(lj1.index[0])
    a_row = pd.Series([0])
    row_df = pd.DataFrame([a_row])
    lj1 = pd.concat([row_df, lj1], ignore_index=True)
    lj1.rename(columns={lj1.columns[0]: 'lj1'}, inplace=True)
    pos = len(lj1) - 1
    # Delete the last one
    lj1 = lj1.drop(lj1.index[pos])

    df_lj_12 = pd.concat([lj, lj1], axis=1)
    df_lj_12.rename(columns={df_lj_12.columns[0]: 'lj'}, inplace=True)

    x2j = df_lj_12['lj'] - df_lj_12['lj1']
    df_x2j = pd.DataFrame(x2j)

    return df_x2j


def pre_RR_interval(df_merge_col, string_ecg):
    """
            pre-RR Interval.

                    Parameters
                    ----------
                    df_merge_col : DataFrame Pandas
                    string_ecg : String. Ex 'ECG_P_Peaks'

                    Returns
                    -------
                    dataFrame : Pandas dataframe column with data peaks interval requested.

                  """

    # pre-RR interval
    lj1 = extract_feature(df_merge_col, string_ecg)
    # lj1 = lj1.drop(lj1.index[0])
    a_row = pd.Series([0])
    row_df = pd.DataFrame([a_row])
    lj1 = pd.concat([row_df, lj1], ignore_index=True)
    lj1.rename(columns={lj1.columns[0]: 'lj1'}, inplace=True)
    pos = len(lj1) - 1
    # Delete the last one
    lj1 = lj1.drop(lj1.index[pos])

    lj2 = extract_feature(df_merge_col, string_ecg)

    a_row = pd.Series([0])
    row_df = pd.DataFrame([a_row])
    lj2 = pd.concat([row_df, row_df, lj2], ignore_index=True)
    lj2.rename(columns={lj2.columns[0]: 'lj2'}, inplace=True)

    # Delete the last two
    lj2 = lj2.drop(lj2.index[len(lj2) - 1])
    lj2 = lj2.drop(lj2.index[len(lj2) - 1])

    df_lj_12 = pd.concat([lj1, lj2], axis=1)

    x2j = interval(df_lj_12, 'lj2', 'lj1')
    return x2j


def pos_RR_interval(df_merge_col, string_ecg):
    """
            pos-RR Interval.

                    Parameters
                    ----------
                    df_merge_col : DataFrame Pandas
                    string_ecg : String. Ex 'ECG_P_Peaks'

                    Returns
                    -------
                    dataFrame : Pandas dataframe column with data peaks interval requested.

                  """

    # pos-RR interval

    lj3 = extract_feature(df_merge_col, string_ecg)
    lj3 = lj3.drop(lj3.index[0])
    a_row = pd.Series([0])
    row_df = pd.DataFrame([a_row])
    lj3 = pd.concat([lj3, row_df], ignore_index=True)
    lj3.rename(columns={lj3.columns[0]: 'lj3'}, inplace=True)

    lj4 = extract_feature(df_merge_col, string_ecg)

    df_lj_34 = pd.concat([lj3, lj4], axis=1)
    df_lj_34.rename(columns={df_lj_34.columns[1]: 'lj4'}, inplace=True)

    x3j = interval(df_lj_34, 'lj4', 'lj3')

    return x3j


def extract_next_values(df_merge_col, string_ecg):
    """
            Extract the next i+1 values from dataframe.

                    Parameters
                    ----------
                    df_merge_col : DataFrame Pandas
                    string_ecg : String. Ex 'ECG_P_Peaks'

                    Returns
                    -------
                    dataFrame : Pandas dataframe column with data peaks requested.

                  """

    lj3 = extract_feature(df_merge_col, string_ecg)
    lj3 = lj3.drop(lj3.index[0])
    a_row = pd.Series([0])
    row_df = pd.DataFrame([a_row])
    lj3 = pd.concat([lj3, row_df], ignore_index=True)

    new_column_name = string_ecg + '1'
    lj3.columns = [new_column_name]

    # lj3.rename(columns={lj3.columns[0]: string_ecg + 1}, inplace=True)

    # lj4 = extract_feature(df_merge_col, string_ecg)

    # df_lj_34 = pd.concat([lj3, lj4], axis=1)
    # df_lj_34.rename(columns={df_lj_34.columns[1]: 'lj4'}, inplace=True)

    x3j = lj3  # lj3[string_ecg + 1]

    return x3j


# ------> Preparing data
# Retrieve ECG data from data folder (sampling rate= 100 Hz)
data = pd.read_csv(DATASET)

# Process ecg
ecg_signals, info = nk.ecg_process(data[ECG_COLUMN_NAME], sampling_rate=FREQUENCY)

# Clean an ECG signal.
# ecg_cleaned = nk.ecg_clean(ecg_signals, sampling_rate=FREQUENCY)

# First, we need extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg_signals, sampling_rate=FREQUENCY)

# Later, delineate the ECG signal and visualizing all peaks of ECG complexes. Define show=True
_, waves_peak = nk.ecg_delineate(ecg_signals, rpeaks, sampling_rate=FREQUENCY, show=False, show_type='peaks')

# ------> Utilities
# Plotting 30 seconds (30*100Hz=3000)
# Plotting 600 seconds (600*360Hz=216000)
# Plotting 60 seconds (60*360Hz=21600)
plot = nk.ecg_plot(ecg_signals[:DATAPOINTS], sampling_rate=FREQUENCY)

# ------> Creating DataFrame Pandas
dataPeaks = pd.DataFrame(waves_peak)
dataPeaks.index.name = 'Cardiac Cycle (beat)'

r_peaks = rpeaks['ECG_R_Peaks']
dataPeaks_r = pd.DataFrame.from_dict(r_peaks, orient='columns')
dataPeaks_r.columns = ['ECG_R_Peaks']

# Merge R Peaks on Pandas DataFrame
dataPeaks = pd.concat([dataPeaks, dataPeaks_r], axis=1)

# Delete the first and last record.
# dataPeaks = dataPeaks[1:-1]

# df_merge_col = pd.DataFrame()

#################################
# In the first category of primary features, the (x, y) of P,Q,R,S, and T are extracted
# for each segment. For example, Px and Py refers to x and y coordinate of the P wave
#################################

# -->Peaks(P,Q,R,S,T)
p_peaks = extract_feature(dataPeaks, 'ECG_P_Peaks')
q_peaks = extract_feature(dataPeaks, 'ECG_Q_Peaks')
r_peaks = extract_feature(dataPeaks, 'ECG_R_Peaks')
s_peaks = extract_feature(dataPeaks, 'ECG_S_Peaks')
t_peaks = extract_feature(dataPeaks, 'ECG_T_Peaks')

peaks_values = pd.concat([p_peaks, q_peaks, r_peaks, s_peaks, t_peaks], axis=1)
df_merge_col = peaks_values

# -->Px/Py values
p_x = extract_x(df_merge_col, 'ECG_P_Peaks', 'ECG_P_Peaks')
p_y = extract_y(ecg_signals['ECG_Raw'], df_merge_col, 'ECG_P_Peaks')
# pxy_values = pd.concat([df_merge_col, p_x, p_y], axis=1)
# df_merge_col = pxy_values

# -->Qx/Qy values
q_x = extract_x(df_merge_col, 'ECG_P_Peaks', 'ECG_Q_Peaks')
q_y = extract_y(ecg_signals['ECG_Raw'], df_merge_col, 'ECG_Q_Peaks')
# qxy_values = pd.concat([df_merge_col, q_x, q_y], axis=1)
# df_merge_col = qxy_values

# -->Rx/Ry values
r_x = extract_x(df_merge_col, 'ECG_P_Peaks', 'ECG_R_Peaks')
r_y = extract_y(ecg_signals['ECG_Raw'], df_merge_col, 'ECG_R_Peaks')
# rxy_values = pd.concat([df_merge_col, r_x, r_y], axis=1)
# df_merge_col = rxy_values

# -->Sx/Sy values
s_x = extract_x(df_merge_col, 'ECG_P_Peaks', 'ECG_S_Peaks')
s_y = extract_y(ecg_signals['ECG_Raw'], df_merge_col, 'ECG_S_Peaks')
# sxy_values = pd.concat([df_merge_col, s_x, s_y], axis=1)
# df_merge_col = sxy_values

# -->Tx/Ty values
t_x = extract_x(df_merge_col, 'ECG_P_Peaks', 'ECG_T_Peaks')
t_y = extract_y(ecg_signals['ECG_Raw'], df_merge_col, 'ECG_T_Peaks')
# txy_values = pd.concat([df_merge_col, t_x, t_y], axis=1)
# df_merge_col = txy_values

xy_values = pd.concat([df_merge_col, p_x, p_y, q_x, q_y, r_x, r_y, s_x, s_y, t_x, t_y], axis=1)
df_merge_col = xy_values

#################################
# In the second category of primary features, notation OnQRS_x refers to the average of x values of P and
# Q waves and the average of x values of S and T is defined as OffQRS_x.
# Notations OnQRS_y and OnQRS_y are the corresponding values of ECG signal for OnQRS_x and OffQRS_x, respectively.
#################################

# -->On/Off QRS_x
onQRS_x = extract_QRS_x(df_merge_col, 'Px', 'Qx', 'On')
offQRS_x = extract_QRS_x(df_merge_col, 'Sx', 'Tx', 'Off')

df_QRS_x = pd.concat([df_merge_col, onQRS_x, offQRS_x], axis=1)
df_merge_col = df_QRS_x

# -->On/Off QRS_y
onQRS_y = extract_QRS_y(ecg_signals['ECG_Raw'], df_merge_col, 'OnQRS_x', 'On')
offQRS_y = extract_QRS_y(ecg_signals['ECG_Raw'], df_merge_col, 'OffQRS_x', 'Off')

df_QRS_y = pd.concat([df_merge_col, onQRS_y, offQRS_y], axis=1)
df_merge_col = df_QRS_y

#################################
# In the third category of primary features, the interval (difference between the x values of two waves) are measured.
# For example, PPinterval means the difference between the x value of the P of the current segment
# and the x value of the P of the next segment.
# RR2 interval means the distance between the x value of R of the current segment
# and x value of R-peak of the next-but-one segment.
#################################

# ---> Interval same

pp_interval = interval_same(df_merge_col, 'Px')
qq_interval = interval_same(df_merge_col, 'Qx')
rr_interval = interval_same(df_merge_col, 'Rx')
ss_interval = interval_same(df_merge_col, 'Sx')
tt_interval = interval_same(df_merge_col, 'Tx')

intervals = pd.concat([df_merge_col, pp_interval, qq_interval, rr_interval, ss_interval, tt_interval], axis=1)
df_merge_col = intervals

# ---> Interval2

pp2_interval = interval_2(df_merge_col, 'Px', 'ECG_P_Peaks')
qq2_interval = interval_2(df_merge_col, 'Qx', 'ECG_Q_Peaks')
rr2_interval = interval_2(df_merge_col, 'Rx', 'ECG_R_Peaks')
ss2_interval = interval_2(df_merge_col, 'Sx', 'ECG_S_Peaks')
tt2_interval = interval_2(df_merge_col, 'Tx', 'ECG_T_Peaks')

intervals2 = pd.concat([df_merge_col, pp2_interval, qq2_interval, rr2_interval, ss2_interval, tt2_interval], axis=1)
df_merge_col = intervals2

#################################
# In the fourth category of primary features, the difference of the y-values of two peaks are measured.
# For example, R − Ramplitude means the difference between the y value of the R-peak of the current segment
# and the y value of the R-peak of the next segment.
# R − R2amplitude means the difference between the y value of R-peak of the current segment and
# the R-peak of the next-but-one segment.
#################################
ecg_signals_raw = ecg_signals['ECG_Raw']

pp_amplitude = amplitude_(ecg_signals_raw, df_merge_col, 'ECG_P_Peaks')
qq_amplitude = amplitude_(ecg_signals_raw, df_merge_col, 'ECG_Q_Peaks')
rr_amplitude = amplitude_(ecg_signals_raw, df_merge_col, 'ECG_R_Peaks')
ss_amplitude = amplitude_(ecg_signals_raw, df_merge_col, 'ECG_S_Peaks')
tt_amplitude = amplitude_(ecg_signals_raw, df_merge_col, 'ECG_T_Peaks')

amplitudes = pd.concat([df_merge_col, pp_amplitude, qq_amplitude, rr_amplitude, ss_amplitude, tt_amplitude], axis=1)
df_merge_col = amplitudes

pp_amplitude2 = amplitude_2(ecg_signals_raw, df_merge_col, 'ECG_R_Peaks')
qq_amplitude2 = amplitude_2(ecg_signals_raw, df_merge_col, 'ECG_Q_Peaks')
rr_amplitude2 = amplitude_2(ecg_signals_raw, df_merge_col, 'ECG_R_Peaks')
ss_amplitude2 = amplitude_2(ecg_signals_raw, df_merge_col, 'ECG_S_Peaks')
tt_amplitude2 = amplitude_2(ecg_signals_raw, df_merge_col, 'ECG_T_Peaks')

amplitudes2 = pd.concat([df_merge_col, pp_amplitude2, qq_amplitude2, rr_amplitude2, ss_amplitude2, tt_amplitude2],
                        axis=1)
df_merge_col = amplitudes2

#################################
# Features from Antonio's e-mail.
#################################

# ----> Intervals
# PR_Interval = time
pr_interval = interval(df_merge_col, 'ECG_P_Peaks', 'ECG_R_Peaks')
pr_interval.rename(columns={pr_interval.columns[2]: 'P->R interval'}, inplace=True)
pr_interval = pr_interval['P->R interval']
# pr_interval_norm = normalize_column(pr_interval['P->R interval(miliseconds)'])

# QRS_Interval = time
qrs_interval = interval(df_merge_col, 'ECG_Q_Peaks', 'ECG_S_Peaks')
qrs_interval.rename(columns={qrs_interval.columns[2]: 'Q->S interval'}, inplace=True)
qrs_interval = qrs_interval['Q->S interval']

# qrs_interval_norm = normalize_column(qrs_interval['Q->S interval(miliseconds)'])

# QT_interval = time
qt_interval = interval(df_merge_col, 'ECG_Q_Peaks', 'ECG_T_Peaks')
qt_interval.rename(columns={qt_interval.columns[2]: 'Q->T interval'}, inplace=True)
qt_interval = qt_interval['Q->T interval']
# qt_interval_norm = normalize_column(qt_interval['Q->T interval(miliseconds)'])

# TQ_interval = time
# TQ = Qnext - Tcurrent
t_current = extract_feature(df_merge_col, 'ECG_T_Peaks')
q_next = extract_next_values(df_merge_col, 'ECG_Q_Peaks')
df_tq = pd.concat([t_current, q_next], axis=1)
tq_interval = interval(df_tq, 'ECG_T_Peaks', 'ECG_Q_Peaks1')
tq_interval.rename(columns={tq_interval.columns[2]: 'T->Q interval'}, inplace=True)
tq_interval = tq_interval['T->Q interval']
# tq_interval_norm = normalize_column(tq_interval['T->Q interval(miliseconds)'])

intervals_ = pd.concat([df_merge_col, pr_interval, qrs_interval, qt_interval, tq_interval], axis=1)
df_merge_col = intervals_

# intervals_ = pd.concat([df_merge_col, pr_interval_norm, qrs_interval_norm, qt_interval_norm, tq_interval_norm], axis=1)
# df_merge_col = intervals_

# ---->Segments

# PR_Segment = time
# PR_Segment = time (P Peak & R Offset)
pr_segment = interval(df_merge_col, 'ECG_P_Peaks', 'ECG_R_Peaks')
pr_segment.rename(columns={pr_segment.columns[2]: 'P->R segment'}, inplace=True)
pr_segment = pr_segment['P->R segment']

# pr_segment_norm = normalize_column(pr_segment['P->R segment(miliseconds)'])

# ST_Segment = time
# ST_Segment = time (S Peak & T Offset)
st_segment = interval(df_merge_col, 'ECG_S_Peaks', 'ECG_T_Peaks')
st_segment.rename(columns={st_segment.columns[2]: 'S->T segment'}, inplace=True)
st_segment = st_segment['S->T segment']
# st_segment_norm = normalize_column(st_segment['S->T segment(miliseconds)'])

segments_ = pd.concat([df_merge_col, pr_segment, st_segment], axis=1)
df_merge_col = segments_
# segments_ = pd.concat([df_merge_col, pr_segment_norm, st_segment_norm], axis=1)
# df_merge_col = segments_

'''
 Amplitudes (refers to the amplitude)

  P_Y, R_Y, Q_Y, S_Y, T_Y
'''

# Solved

'''
Durations

  P, Q, R, S, T

Y-differences

  PQ_y_diff,  QR_y_diff, RS_y_diff, ST_y_diff
'''

pq_y_diff = interval(df_merge_col, 'Py', 'Qy')
pq_y_diff.rename(columns={pq_y_diff.columns[2]: 'PQ_y_diff'}, inplace=True)
pq_y_diff = pq_y_diff['PQ_y_diff']

qr_y_diff = interval(df_merge_col, 'Qy', 'Ry')
qr_y_diff.rename(columns={qr_y_diff.columns[2]: 'QR_y_diff'}, inplace=True)
qr_y_diff = qr_y_diff['QR_y_diff']

rs_y_diff = interval(df_merge_col, 'Ry', 'Sy')
rs_y_diff.rename(columns={rs_y_diff.columns[2]: 'RS_y_diff'}, inplace=True)
rs_y_diff = rs_y_diff['RS_y_diff']

st_y_diff = interval(df_merge_col, 'Sy', 'Ty')
st_y_diff.rename(columns={st_y_diff.columns[2]: 'ST_y_diff'}, inplace=True)
st_y_diff = st_y_diff['ST_y_diff']

y_diff_ = pd.concat(
    [df_merge_col, pq_y_diff, qr_y_diff, rs_y_diff, st_y_diff], axis=1)
df_merge_col = y_diff_

'''
Inter segments

   onQRS_x = average of x values of P and Q waves

   offQRS_x = average of x values of S and T waves

   onQRS_y = corresponding values of ECG signals (mean P and Q)

   offQRS_y = corresponding values of ECG sginals (mean S and T).

'''

# Solved

'''
Inter heartbeat

     [PP, QQ, RR, SS, TT]_interval

     [PP, QQ, RR, SS, TT]_amplitude (y differences)

     [PP2, QQ2, RR2, SS2, TT2]_interval (next-but-one beat)

     [PP2, QQ2, RR, SS2, TT2]_amplitude (y differences) (next-but-one beat)

'''
# Solved


#################################
# Paper: Unsupervised feature relevance analysis applied to improve ECG heartbeat clustering
#################################
#################################
# Heart rate variability (HRV) derived features (x1j, x2j, x3j). These features are computed as in [29]:
# x1j = lj − lj−1, (RR interval)
# x2j = lj−1 − lj−2, (pre-RR interval)
# x3j = lj+1 − lj, (post-RR interval)
# where lj accounts for the location of the R-wave of the j-th heartbeat.
#################################

# RR interval
x1j = RR_interval(df_merge_col, 'ECG_R_Peaks')
x1j.rename(columns={x1j.columns[0]: 'HRV-RRinterval'}, inplace=True)
x1j = x1j['HRV-RRinterval']

# pre-RR interval
x2j = pre_RR_interval(df_merge_col, 'ECG_R_Peaks')
x2j.rename(columns={x2j.columns[2]: 'HRV-PreRRinterval'}, inplace=True)
x2j = x2j['HRV-PreRRinterval']

# post-RR interval
x3j = pos_RR_interval(df_merge_col, 'ECG_R_Peaks')
x3j.rename(columns={x3j.columns[2]: 'HRV-PosRRinterval'}, inplace=True)
x3j = x3j['HRV-PosRRinterval']

x123_ = pd.concat([df_merge_col, x1j, x2j, x3j], axis=1)
df_merge_col = x123_

#################################
# Heartbeat prematurity features (HPF) (x4j, x5j, x6j): Prematurity features account for changes in heart rate [30],
# and contribute to the identification of S beats. The two first features, # x4j and x5j, can be readily
# calculated as x4j = x1j − x2j and  x5j = x3j − x1j.
# Not only is the absolute value of these features informative, but also their sign changes
#################################

# x4j= x1j - x2j
x4j = interval(df_merge_col, 'HRV-PreRRinterval', 'HRV-RRinterval')
x4j.rename(columns={x4j.columns[2]: 'HPF X4'}, inplace=True)
x4j = x4j['HPF X4']

# x5j = x3j - x1j
x5j = interval(df_merge_col, 'HRV-RRinterval', 'HRV-PosRRinterval')
x5j.rename(columns={x5j.columns[2]: 'HPF X5'}, inplace=True)
x5j = x5j['HPF X5']

x45_ = pd.concat([df_merge_col, x4j, x5j], axis=1)
df_merge_col = x45_

# TODO x6j = 55

# Preview on console
df_merge_col.index.name = 'Cardiac Cycle (beat)'
df_merge_col.index += 1
print(df_merge_col)

# ------> Normalization process

# Delete the first and last record.
# df_merge_col = df_merge_col[1:-1]

# Normalizar os dados calculados (miliseconds) / media, moda, mediana


# Export to CSV all features, missing value save as NaN
df_merge_col.to_csv(os.getcwd() + '/exported/MIT_FeatureExtraction.csv', na_rep='NaN')

print("END>>>>>>")
