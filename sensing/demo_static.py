import os
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Jingle.scheduler.workload_learners.arima_learner import ARIMATSModel

DATA_PATH = 'traces'
SENSING_FILE = 'raw_sensing.csv'
arima_orders = [(1, 1, 1)]


def detect_frequent_activity(df, timestamp, threshold=3, time_window=2, activity_type="entry"):
    """ Detect frequent activity (entry or exit) within a given time window. """
    current_timestamp = pd.to_datetime(timestamp)
    past_timestamp = current_timestamp - pd.Timedelta(minutes=time_window)
    subset_df = df[(df['timestamp'] >= past_timestamp) & (df['timestamp'] <= current_timestamp)]

    frequency = subset_df.shape[0]
    # Assume that positive occupancy changes are entries and negative are exits
    if activity_type == "entry":
        activity_freq = len(subset_df[subset_df['occupancy'].diff() > 0])
    else:
        activity_freq = len(subset_df[subset_df['occupancy'].diff() < 0])

    if activity_freq >= threshold:
        return True
    return False


def main():
    """ Main function. """
    csv_files = [elem for elem in os.listdir(DATA_PATH) if elem.endswith('csv')]
    timeseries_data = {}
    overestimate_factor = 1.7  # Overestimate by 50% after detecting frequent entries
    underestimate_factor = 0.8  # Underestimate by 20% after detecting frequent exits

    # Load raw sensing data for occupancy detection
    sensing_data = pd.read_csv(SENSING_FILE)
    sensing_data['timestamp'] = pd.to_datetime(sensing_data['timestamp'])
    sensing_data['occupancy_diff'] = sensing_data['occupancy'].diff()

    for csvf in csv_files:
        file_path = os.path.join(DATA_PATH, csvf)
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert to datetime
        timeseries_data[csvf] = df

    conf_alpha = 0.9
    start_at = 2
    max_num_tests = 122
    ground_truth = []
    esstimate = []
    scaling_down = False

    for csvf, df in timeseries_data.items():
        orig_time_series = list(df.loc[:, 'load'])
        len_time_series = len(orig_time_series)
        for ao in arima_orders:
            print('Model %s' % (str(ao)))
            start_time = df['timestamp'].iloc[0]

            for pred_idx in range(start_at + 1, min(start_at + 1 + max_num_tests, len_time_series - 1)):
                model = ARIMATSModel(str(ao), ao)
                model.initialise_model()
                history = orig_time_series[:pred_idx]
                model.update_model_with_new_data(history)
                mean_pred, lcb, ucb = model.forecast(num_steps_ahead=1, conf_alpha=conf_alpha)

                current_timestamp = df['timestamp'].iloc[pred_idx]

                if current_timestamp == (start_time + pd.Timedelta(minutes=30)):
                    # Apply adjustment to mean_pred
                    mean_pred = mean_pred* 1.7

                true_val = orig_time_series[pred_idx]
                ground_truth.append(true_val)
                esstimate.append(mean_pred)
    print(esstimate)
    plt.figure().set_figwidth(20)
    plt.plot(ground_truth, color='g', linewidth=2, label='ground truth', marker='o')
    plt.plot(esstimate, color='red', linewidth=2, label='load prediction', marker='o')
    plt.xlabel("time", fontsize=10)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
