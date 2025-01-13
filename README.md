# 🫀 ECG-Based Sleep Monitoring and Time-Domain HRV Analysis

## 📋 **Project Overview**
This project is a comprehensive solution for processing and analyzing **ECG (Electrocardiogram) data** from **EDF (European Data Format) files**. The primary focus is to extract, preprocess, and analyze time-domain **Heart Rate Variability (HRV)** metrics and perform **sleep stage labeling**. The project includes several Python scripts and Jupyter notebooks designed to handle various tasks such as data conversion, windowing, filtering, and feature extraction.

---

## 📁 **Repository Structure**

```
├── ECG + windowed.py         # Converts EDF files to DataFrames and splits them into 30-second windows.
├── Time_domain_main.py       # Comprehensive script for time-domain HRV analysis and sleep stage labeling.
├── gptModel.ipynb            # Jupyter notebook for machine learning model processing (details TBD).
├── dataframeFormatting.ipynb # Jupyter notebook for formatting and preprocessing DataFrames.
├── txt2csv.ipynb             # Converts text-based sleep stage files into CSV format.
├── edf2csv.ipynb             # Converts EDF files into CSV format for easier data handling.
└── README.md                 # Project documentation.
```

---

## 🧪 **Project Features**

### ✅ ECG Signal Processing
- **EDF to DataFrame Conversion**: Efficiently read multi-channel EDF files and convert them to pandas DataFrames.
- **Windowing**: Split ECG data into 30-second windows for better handling and analysis.

### ✅ Time-Domain HRV Analysis
- **Bandpass Filtering**: Apply bandpass filters to remove noise from ECG signals.
- **Z-Score Normalization**: Standardize the ECG signal for further analysis.
- **R-Peak Detection**: Identify R-peaks in ECG signals to calculate **R-R Intervals (RRI)**.
- **HRV Metrics**:
  - **AVNN**: Average of R-R intervals.
  - **SDNN**: Standard deviation of R-R intervals.
  - **RMSSD**: Root mean square of successive differences between R-R intervals.
  - **pNN50**: Percentage of R-R intervals exceeding 50ms.

### ✅ Sleep Stage Labeling
- **Sleep Stage Text Files**: Read sleep stage annotations from text files.
- **Labeling**: Combine the ECG data with sleep stage labels to enable stage-based analysis.

---

## ⚙️ **Setup and Installation**

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required Libraries:
  ```bash
  pip install pandas numpy matplotlib scipy neurokit2 pyedflib heartpy tqdm
  ```

### Cloning the Repository
```bash
git clone https://github.com/yourusername/ecg-sleep-analysis.git
cd ecg-sleep-analysis
```

### Running the Scripts
1. **Convert EDF to DataFrame and Window**
   ```bash
   python "ECG + windowed.py"
   ```

2. **Time-Domain HRV Analysis**
   ```bash
   python "Time_domain_main.py"
   ```

---

## 📊 **HRV Metrics Explained**
| Metric   | Description                                           |
|----------|-------------------------------------------------------|
| AVNN     | Average R-R interval (ms)                             |
| SDNN     | Standard deviation of R-R intervals (ms)              |
| RMSSD    | Root mean square of successive differences (ms)       |
| pNN50    | Percentage of successive RRIs exceeding 50 ms         |
| MeanHR   | Average heart rate (beats per minute)                 |
| MinHR    | Minimum heart rate recorded                           |
| MaxHR    | Maximum heart rate recorded                           |

---

## 📈 **Visualization**
The project includes various plots to visualize the ECG signal, R-peaks, and R-R intervals. These plots help in understanding the behavior of the ECG signal and HRV metrics.

---

## 🤖 **Future Enhancements**
- Integration with **machine learning models** to classify sleep stages.
- Real-time ECG processing.
- Advanced feature extraction using frequency-domain and non-linear HRV metrics.

---

Happy Analyzing! 🫀📊

