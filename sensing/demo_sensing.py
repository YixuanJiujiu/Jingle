import pandas as pd
import matplotlib.pyplot as plt

# Reading datasets from CSV files
df1 = pd.read_csv('traces/time_request.csv', parse_dates=['timestamp'])
df2 = pd.read_csv('raw_sensing.csv', parse_dates=['timestamp'])

# Plotting
fig, ax1 = plt.subplots(figsize=(12,6))

# Plot Dataset 1 on primary y-axis
ax1.plot(df1['timestamp'], df1['load'], color='blue', label='Number of Concurrent Requests (Traces Dataset)')
ax1.set_ylabel('Number of Concurrent Requests', color='blue')
ax1.set_xlabel('Timestamp')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')

# Create a secondary y-axis for Dataset 2
ax2 = ax1.twinx()
ax2.plot(df2['timestamp'], df2['occupancy'], color='red', label='Occupancy (Sensing Data')
ax2.set_ylabel('Occupancy', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right')

plt.title('Time Series Plot')
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
