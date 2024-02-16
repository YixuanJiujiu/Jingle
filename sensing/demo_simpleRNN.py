import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


# Sample data
df = pd.read_csv("time_request.csv", header=None, names=['timestamp', 'value'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Normalize the 'value' column
scaler = MinMaxScaler(feature_range=(0, 1))
df['normalized_value'] = scaler.fit_transform(df['value'].values.reshape(-1, 1))

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        X.append(seq[:-1])
        y.append(seq[-1])
    return np.array(X), np.array(y)

seq_length = 10

# 2. Define SimpleRNN model
model = Sequential()
model.add(SimpleRNN(units=50, activation="tanh", input_shape=(seq_length-1, 1)))
model.add(Dense(units=1))
model.compile(optimizer="adam", loss="mean_squared_error")

# 3. Split dataset, train, and predict
train_size = int(len(df) * 0.7)
train, test = df[0:train_size], df[train_size:]

X_train, y_train = create_sequences(train["normalized_value"].values, seq_length)
X_test, y_test = create_sequences(test["normalized_value"].values, seq_length)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

y_pred = model.predict(X_test)
y_pred_original = scaler.inverse_transform(y_pred)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = sqrt(mean_squared_error(y_test_original, y_pred_original))
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

list = np.concatenate(np.array([-1] * train_size) , y_pred_original.flatten())
print( y_pred_original.flatten())
# 4. Plotting
plt.figure(figsize=(15,6))
plt.plot(y_test_original, label="Ground Truth", color="g",  marker='o')
plt.plot(y_pred_original, label="Predictions", color="red", alpha=0.7,  marker='o')
plt.title("RNN Predictions vs Ground Truth")
plt.xlabel("Time")
plt.ylabel("Load")
plt.legend()
plt.show()



