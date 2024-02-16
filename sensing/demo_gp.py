import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

DATA_PATH = 'traces'


def main():
    """ Main function. """
    csv_files = [elem for elem in os.listdir(DATA_PATH) if elem.endswith('csv')]

    for csvf in csv_files:
        file_path = os.path.join(DATA_PATH, csvf)
        df = pd.read_csv(file_path)

        # Convert 'timestamp' column to UNIX timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10 ** 9
        timestamps = df['timestamp'].values.reshape(-1, 1)
        loads = df['load'].values

        # Define the kernel: Constant * Matern + White Noise
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1)

        # Create the Gaussian Process Regressor model
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # Fit the model
        gp.fit(timestamps, loads)

        # Forecast using the GP model
        forecast_points = np.linspace(timestamps.min(), timestamps.max(), len(timestamps)).reshape(-1, 1)
        y_pred, y_std = gp.predict(forecast_points, return_std=True)

        print(forecast_points)
        # Plotting the actual data and predictions
        plt.figure().set_figwidth(20)
        plt.plot(timestamps, loads, 'g', markersize=10, label='Observations')
        plt.plot(forecast_points, y_pred, 'r', lw=1, label='Prediction')
        plt.fill_between(forecast_points.ravel(), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, alpha=0.2, color='k')

        plt.xlabel("Timestamp")
        plt.ylabel("Load")
        plt.title(f"Time Series Forecasting using GP for {csvf}")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
