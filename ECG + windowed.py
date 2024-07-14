import pyedflib
import pandas as pd

def edf_to_dataframe(edf_file, channels_to_exclude):
    # Open the EDF file
    f = pyedflib.EdfReader(edf_file)
    
    # Get the number of signals
    n_signals = f.signals_in_file
    
    # Get the signal labels
    signal_labels = f.getSignalLabels()
    
    # Initialize an empty dictionary to store signal data
    data_dict = {}
    
    # Read the signals data
    for i in range(n_signals):
        if i < len(signal_labels):
            # Check if the signal label is not in the exclusion list
            if signal_labels[i] not in channels_to_exclude:
                data_dict[signal_labels[i]] = f.readSignal(i)
    
    # Close the EDF file
    f.close()
    
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data_dict)
    
    return df


# Example usage
edf_file = 'SN002.edf'
channels_to_exclude = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1","EEG C3-M2", "EMG chin", "EOG E1-M2", "EOG E2-M2"]
df = edf_to_dataframe(edf_file, channels_to_exclude)


data_record_duration = 25693
fs = len(df) / data_record_duration

window_duration = 30
window_size = int(fs * window_duration)

num_windows = len(df) // window_size

windows = []
for i in range(num_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        window = df.iloc[start_idx : end_idx]
        windows.append(window)
        
print(windows[400])
        
        
        
        