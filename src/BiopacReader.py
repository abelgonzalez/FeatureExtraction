# BioPack ECG Reader solution
import pandas as pd
import os
import numpy as np

WFDB_FILE = os.getcwd() + '\\data\\BH_Telemetry.txt'
df = pd.read_csv(WFDB_FILE)

df_aux = df[36:]
df = df_aux

split_data = df["BH_Telemetry_1hour.acq"].str.split("\t")
split_data = split_data.replace(r'^\s+$', np.nan, regex=True)

data_list = split_data.to_list()
column_names = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8", "Ch9", "Ch10",
                "Ch11", "Ch12", "Ch13", "Ch14", "Ch15", "Ch16"]

new_df = pd.DataFrame(data_list, columns=column_names)
# delete the last column because it is empty
new_df = new_df.drop(labels='Ch16', axis=1)

df = new_df
print(df)

df.to_csv(os.getcwd() + '/exported/BioPac.csv', na_rep='NaN')
