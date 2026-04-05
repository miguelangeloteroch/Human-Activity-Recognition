# Activity Detector — HAR Android App

Android application for real-time Human Activity Recognition (HAR) using on-device machine learning.

## Overview

The app reads accelerometer data from the device sensor, runs inference locally via a TFLite model, and classifies the user's current activity into three categories: **walking**, **jogging**, and **stationary**.
At the end of each session it shows a summary with time and percentage breakdown per activity.

## Android App

- **Language:** Kotlin
- **UI:** Jetpack Compose
- **Architecture:** MVVM with Repository pattern
- **ML inference:** TensorFlow Lite (on-device, no network required)
- **Sensors:** Android accelerometer via `SensorManager`, exposed as a Kotlin `Flow`
- **Persistence:** Room (user accounts) + DataStore (session preferences)
- **Decision logic:** Majority-vote window over the last N predictions, with confidence thresholds and hysteresis to avoid noisy label switching

## Model Training

The LSTM models were trained on the [WISDM dataset](https://www.cis.fordham.edu/wisdm/dataset.php) using TensorFlow/Keras. The pipeline includes:

- Data cleaning and label normalization
- Winsorization of outliers and iterative imputation of missing values
- Sliding-window segmentation for LSTM input
- Hyperparameter grid search with stratified k-fold cross-validation
- Export to TFLite for on-device deployment

**Requirements:** Python 3.9+, TensorFlow 2.x, scikit-learn, pandas, numpy, matplotlib, seaborn
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
python ficheros_entreno_modelo/wisdm_lstm_pipeline.py --data-path <path/to/WISDM_raw.txt>
```

## Recognized Activities

| Label        | Description              |
|--------------|--------------------------|
| `walking`    | Normal walking pace /stairs|
| `jogging`    | Running / jogging        |
| `stationary` | Standing still or seated |
