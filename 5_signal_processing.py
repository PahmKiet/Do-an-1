import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt

def run_processing(filename):
    RAW_FOLDER, FILTERED_FOLDER = "collected_data", "filtered_data"
    if not os.path.exists(FILTERED_FOLDER): os.makedirs(FILTERED_FOLDER)

    df = pd.read_csv(os.path.join(RAW_FOLDER, filename))
    plt.figure(figsize=(10, 4))
    plt.plot(df['Time'], df['Gap_mm'], color='gray', alpha=0.6, label='Raw')
    plt.title(f"Raw Signal: {filename}")
    plt.savefig(os.path.join(RAW_FOLDER, filename.replace(".csv", "_RAW_PLOT.png")))
    plt.close()

    gap_median = medfilt(df['Gap_mm'].values, kernel_size=5)
    fs = 1 / (df['Time'].iloc[1] - df['Time'].iloc[0]) if len(df) > 1 else 30
    b, a = butter(5, 6/(0.5 * fs), btype='low')
    df['Gap_mm'] = np.round(filtfilt(b, a, gap_median), 2)
    df.to_csv(os.path.join(FILTERED_FOLDER, filename), index=False)
    return True