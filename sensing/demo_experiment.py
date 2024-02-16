import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def calculate_psnr(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(actual)
    psnr = 10 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_csi_advanced(actual, predicted, threshold, lead_time=3):
    hits = 0
    misses = 0
    false_alarms = 0
    for i in range(lead_time, len(actual)):
        if actual[i] >= threshold and np.all(actual[i-lead_time:i] < threshold):
            if np.any(predicted[i-lead_time:i] >= threshold):
                hits += 1
            else:
                misses += 1
        # elif np.any(predicted[i-lead_time:i] >= threshold):
        #     false_alarms += 1
    print(hits, misses, false_alarms)
    csi = hits / float(hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
    return csi

# File names
# files = ['sarima_res.csv', 'arima_20.csv', 'static.csv', 'rnn_res.csv', 'ma_res.csv']
files = ['ma_res.csv', 'arima_20.csv', 'rnn_res.csv', 'static.csv', 'sarima_res.csv']
Model = ['MovingAvg', 'ARIMA', 'RNN','ARIMA-static', 'ARIMA-IoT']

# files = ['arima_10.csv','arima_20.csv', 'arima_50.csv', 'arima_100.csv']
# Model = ['ARIMA-10', 'ARIMA-20', 'ARIMA-50','ARIMA-100']

# Create a figure and axis for plotting
plt.figure(figsize=(20, 5))
df = pd.read_csv("time_request.csv", header=None, names=['timestamp', 'value'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
length = len(df['value'])
time = [i * 30 for i in range(length)]
plt.plot(time, df['value'], label='ground truth', marker = 'o')

ground_truth = np.array( df['value'] )

index= 0
# Loop through each file
for file in files:
    # Read the CSV file
    data = pd.read_csv(file, header=None).iloc[0, :]
    # Plot the data using the index as the x-axis
    plt.plot(time, data[:length], label=Model[index], marker = 'o')  # Assuming the values are in the first column
    data = np.array(data[:122])
    mae = mean_absolute_error(ground_truth, data)
    # psnr = calculate_psnr(ground_truth, data)
    csi = calculate_csi_advanced(ground_truth, data, threshold=9)
    print("Mean Absolute Error (MAE):", mae)
    # print("psnr:", psnr)
    print("csi:", csi)
    index += 1
# Add legend and labels
plt.legend(fontsize=24, loc='upper left', ncol=2)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Time (s)', fontsize=24)
plt.ylabel('Load', fontsize=24)
plt.yticks([0, 5, 10, 15])
# plt.title('Time Series Data from Multiple Files')
plt.tight_layout()
# plt.savefig('arima_sim.png', dpi=300)
plt.savefig('workload_sim.png', dpi=300)
# Show the plot
plt.show()
