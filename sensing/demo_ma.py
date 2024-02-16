import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv("time_request.csv", header=None, names=['value'])

# Define the window size for the moving average
window_size = 3

# Initialize the predictions list with the last 'window_size' values from 'data'
predictions = []

# Generate additional predictions
for i in range(len(data) - window_size + 1):
    # Take the most recent 'window_size' values
    window_values = data['value'][i:i+window_size]
    # Calculate the next prediction and append to predictions
    next_prediction = window_values.mean()
    predictions.append(next_prediction)

# The 'predictions' list now includes the original data and the additional 122 predictions
# Print only the additional predictions
print(predictions[-(122-window_size):])
