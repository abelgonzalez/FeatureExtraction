# Load NeuroKit and other useful packages
import os

import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

WFDB_FILE = os.getcwd() + '\\data\\bio_resting_5min_100hz.csv'

data = pd.read_csv(WFDB_FILE)

ecg_signals, info = nk.ecg_process(data["ECG"],
                                   sampling_rate=100)  # sampling_rate (100Hz) in which the signals were generated

plt = nk.ecg_plot(ecg_signals[:3000], sampling_rate=100)  # 3000 datapoints (or 30s) to visualize.

# Extract features
print(nk.ecg_intervalrelated(ecg_signals))

plt.show()
