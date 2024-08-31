# ECG Data Processing and HRV Analysis Pipeline
## Overview

This code processes ECG data stored in EDF (European Data Format) files, applies filtering and normalization, and calculates heart rate variability (HRV) metrics in the time domain. The output is saved as CSV files, which can be used for further analysis or machine learning models.

## Features

- **EDF File Reading**: Converts EDF data into a pandas DataFrame.
- **ECG Signal Transformation**: Reshapes the ECG signal for further processing.
- **Stage Annotation**: Reads and applies sleep stage annotations to the ECG data.
- **Signal Filtering**: Applies a bandpass filter to remove noise from the ECG signal.
- **Z-Score Normalization**: Normalizes the ECG signal for consistent analysis.
- **R-Peak Detection**: Detects R-peaks in the ECG signal, which are crucial for calculating HRV metrics.
- **HRV Metrics Calculation**: Computes standard time-domain HRV metrics like AVNN, SDNN, RMSSD, etc.
- **CSV Export**: Saves the computed HRV metrics along with the corresponding sleep stage annotations to CSV files.

## Dependencies

The following Python libraries are required to run the code:

- `heartpy`
- `pyedflib`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `neurokit2`
- `tqdm`
- `os`

You can install them using pip:

```bash
pip install heartpy pyedflib pandas numpy scipy matplotlib neurokit2 tqdm


