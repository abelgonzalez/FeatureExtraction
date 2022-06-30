import pandas as pd
import neurokit2 as nk

ecg = nk.ecg_simulate(duration=10, method="simple")
ecg2 = nk.ecg_simulate(duration=10, method="ecgsyn")
pd.DataFrame({"ECG_Simple": ecg,
            "ECG_Complex": ecg2}).plot(subplots=True)


signals = pd.DataFrame({"ECG_Raw" : ecg,
                       "ECG_NeuroKit" : nk.ecg_clean(ecg, sampling_rate=1000, method="neurokit"),
                       "ECG_BioSPPy" : nk.ecg_clean(ecg, sampling_rate=1000, method="biosppy"),
                       "ECG_PanTompkins" : nk.ecg_clean(ecg, sampling_rate=1000, method="pantompkins1985"),
                       "ECG_Hamilton" : nk.ecg_clean(ecg, sampling_rate=1000, method="hamilton2002"),
                       "ECG_Elgendi" : nk.ecg_clean(ecg, sampling_rate=1000, method="elgendi2010"),
                       "ECG_EngZeeMod" : nk.ecg_clean(ecg, sampling_rate=1000, method="engzeemod2012")})
signals.plot()

#Clean an ECG signal.
#Prepare a raw ECG signal for R-peak detection with the specified method.
#ecg_clean(ecg_signal, sampling_rate=1000, method='neurokit')


#Find R-peaks in an ECG signal.
#ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
signals, info = nk.ecg_peaks(cleaned, correct_artifacts=True)
nk.events_plot(info["ECG_R_Peaks"], cleaned)


# Segment an ECG signal into single heartbeats.
ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)
ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
nk.ecg_segment(ecg_cleaned, rpeaks=None, sampling_rate=1000, show=True)


#epochs = nk.create_epochs(bio["df"], events["onsets"], duration=700, onset=-100)
