import heartpy as hp
import pyedflib
import pandas as pd
import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import neurokit2 as nk
from tqdm import tqdm
from scipy.interpolate import interp1d

# Function to read EDF file and convert to DataFrame
def edf_to_dataframe(edf_file):
    # Open the EDF file
    f = pyedflib.EdfReader(edf_file)
    
    # Get the number of signals
    n_signals = f.signals_in_file
    
    # Get the signal labels
    signal_labels = f.getSignalLabels()
    
    # Read the signals data
    signals_data = []
    for i in range(n_signals):
        signals_data.append(f.readSignal(i))
    
    # Close the EDF file
    f.close()
    
    # Create a dictionary with the signal labels as keys and the signal data as values
    data_dict = {signal_labels[i]: signals_data[i] for i in range(n_signals)}
    
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data_dict)
    
    return df

# Function to transform EDF data into a new DataFrame with reshaped ECG signal
def transformEdf(df_edf):
    new_df = pd.DataFrame()  # Create an empty DataFrame to store the transformed data
    
    # Loop through each chunk of 7680 rows in the original DataFrame
    for row in tqdm(range(0, df_edf.shape[0], 7680)):
        # Extract the ECG signal from the current chunk and reshape it into a new column
        new_df = pd.concat([new_df, pd.DataFrame({row // 7680: df_edf.loc[row:row+7679, 'ECG'].values})], axis=1)
    
    return new_df

# Function to read stage annotations from a text file and convert them into a DataFrame
def readStage(filename):
    with open(filename) as f:
        data = [line.split(',') for line in f.readlines()]
    df = pd.DataFrame(data)
    df.columns = df.iloc[0].str.strip()  # Set column names based on the first row of data
    df.drop(0, inplace=True)  # Drop the first row (which contains the column names)
    df['Duration'] = df['Duration'].astype(int)  # Convert 'Duration' column to integer type
    return df.drop(df[df['Duration'] < 30].index)  # Remove rows with duration less than 30

# Function to label the transformed EDF data with stage annotations
def labeling(df_trans, df_stage):
    return pd.concat([df_trans.T, pd.DataFrame({"stage": df_stage['Annotation'].values})], axis=1)

filename = 'SN002_sleepscoring.txt'    
edf_file = "SN002.edf"
df_edf = edf_to_dataframe(edf_file)
df_stage = readStage(filename)
df_trans = transformEdf(df_edf)
df_done = labeling(df_trans, df_stage)
df_done.head()
    
#Filtering
def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

#Undergo Z-score normalization
def z_score_normalize(signal):
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    normalized_signal = (signal - mean_val) / std_val
    return normalized_signal

#Empty list for saving time domain HRV result
#Average RRI
AVNN = []
#Std RRI
SDNN = []
MeanHR = []
StdHR = []
MinHR = []
MaxHR = []
#Root mean square of successive differences between normal heartbeats (Sum in ms)
RMSSD = []
#RRI exceeds 50ms (Count)
NN50 = []
#Percentage of RRI exceeds 50ms
pNN50 = []


rounds = len(df_done) // 50
remaining = len(df_done) % 50

#RRI thresold
low_thresold = 550
up_threshold = 1600

#Time domain HRV strat here
for j in range (0,rounds):
    for i in range (0,50):
        sample_rate = 256
        #Apply bandpass filter on ECG data
        filtered = df_done.iloc[i,:-1]
        # Assuming you want to process the first row
        filtered_data = bandpass(filtered, [0.5, 40], sample_rate)
        #Apply Z-score normalization on ECG data
        z_score_normalized_ecg = z_score_normalize(filtered_data)
        
        
        
        #R-peak detection here
        # Process the ECG signal
        wd, m = hp.process(z_score_normalized_ecg, sample_rate=256)
        # Extract R-peaks
        r_peaks = wd['peaklist']
        # Calculate R-R Intervals (RRI)
        rri = np.diff(r_peaks) / sample_rate * 1000  # Convert to milliseconds
        medians_rri = np.median(rri)
        heart_rate = 60000/rri
        
        rri_clean = rri[(rri > low_thresold) & (rri < up_threshold)]
        new_x = np.linspace(0, len(rri_clean), len(rri))
        rri_interp = np.interp(new_x, np.arange(len(rri_clean)), rri_clean)
        
        hr = 60000/rri
        
        #Time domain HRV results here and append to empty list
        AVNN.append(np.mean(rri))
        SDNN.append(np.std(rri))
        MeanHR.append(np.mean(hr))
        StdHR.append(np.std(hr))
        MinHR.append(np.min(hr))
        MaxHR.append(np.max(hr))
        RMSSD.append(np.sqrt(np.mean(np.square(np.diff(rri)))))
        NN50.append(np.sum(np.abs(np.diff(rri)) > 50)*1)
        pNN50.append(100 * np.sum((np.abs(np.diff(rri)) > 50)*1) / len(rri))

for x in range (rounds * 50 ,len(df_done)-1):
    sample_rate = 256
    #Apply bandpass filter on ECG data
    filtered = df_done.iloc[x,:-1]
    # Assuming you want to process the first row
    filtered_data = bandpass(filtered, [0.5, 40], sample_rate)
    #Apply Z-score normalization on ECG data
    z_score_normalized_ecg = z_score_normalize(filtered_data)
    
    
    
    #R-peak detection here
    # Process the ECG signal
    wd, m = hp.process(z_score_normalized_ecg, sample_rate=256)
    # Extract R-peaks
    r_peaks = wd['peaklist']
    # Calculate R-R Intervals (RRI)
    rri = np.diff(r_peaks) / sample_rate * 1000  # Convert to milliseconds
    medians_rri = np.median(rri)
    heart_rate = 60000/rri
    
    rri_clean = rri[(rri > low_thresold) & (rri < up_threshold)]
    new_x = np.linspace(0, len(rri_clean), len(rri))
    rri_interp = np.interp(new_x, np.arange(len(rri_clean)), rri_clean)
    
    hr = 60000/rri
    
    # HRV metrics
    AVNN.append(np.mean(rri))
    SDNN.append(np.std(rri))
    MeanHR.append(np.mean(hr))
    StdHR.append(np.std(hr))
    MinHR.append(np.min(hr))
    MaxHR.append(np.max(hr))
    RMSSD.append(np.sqrt(np.mean(np.square(np.diff(rri)))))
    NN50.append(np.sum(np.abs(np.diff(rri)) > 50)*1)
    pNN50.append(100 * np.sum((np.abs(np.diff(rri)) > 50)*1) / len(rri))
# =============================================================================
# #Normalization and RRI plot
# plt.figure(figsize=(12, 4))
# plt.plot(z_score_normalized_ecg, label='ECG Signal')
# plt.scatter(r_peaks, [z_score_normalized_ecg[i] for i in r_peaks], color='red', label='R-peaks')
# plt.legend()
# plt.show()
# 
# plt.figure(figsize=(12, 4))
# plt.plot(rri, label='R-R Intervals (ms)')
# plt.xlabel('Beat number')
# plt.ylabel('R-R Interval (ms)')
# plt.legend()
# plt.show()
# 
# plt.figure(figsize=(12, 4))
# plt.plot(rri_interp, label='R-R Intervals (ms)')
# plt.xlabel('Beat number')
# plt.ylabel('rri_interp')
# plt.legend()
# plt.show()
# =============================================================================




