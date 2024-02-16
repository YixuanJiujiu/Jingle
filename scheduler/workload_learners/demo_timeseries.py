"""
    A simple demo using time series.

    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
"""

# pylint: disable=too-many-locals

import os
import time
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# Local
from  Jingle.scheduler.workload_learners.arima_learner import ARIMATSModel



DATA_PATH = 'traces'


arima_orders = [
    # (0, 0, 1),
    (1, 1, 1),
    # (5, 1, 0),
    # (1, 1, 2),
#     (2, 1, 2),
#     (0, 0, 5),
#     (3, 0, 2),
    ]



def plot_ts(orig_time_series):
    """ Plots time series. """
    num_orig_data = len(orig_time_series)
    plt.figure()
    plt.plot(range(num_orig_data), orig_time_series)
    plt.show()


def main():
    """ Main function. """
    csv_files = [elem for elem in os.listdir(DATA_PATH) if elem.endswith('csv')]
    timeseries_data = {}
    for csvf in csv_files:
        file_path = os.path.join(DATA_PATH, csvf)
        df = pd.read_csv(file_path)
        timeseries_data[csvf] = list(df.loc[:, 'load'])
        #print(timeseries_data[csvf])

    conf_alpha = 0.2
    start_at = 1
    max_num_tests = 40
    ground_truth = []
    esstimate = []
    esstimate2 = []
    last_pred = 0
    for csvf, orig_time_series in timeseries_data.items():
        len_time_series = len(orig_time_series)
        errors = {}
        trapped_by_ci = {}
        avg_times = {}
        conf_widths = {}
        # Create model
        for ao in arima_orders:
            curr_errs = []
            curr_trapped_by_ci = []
            curr_conf_widths = []
            curr_times = []
            print('Model %s'%(str(ao)))
            for pred_idx in range(start_at + 1,
                                  min(start_at + 1 + max_num_tests, len_time_series - 1)):
                start_time = time.time()
                model = ARIMATSModel(str(ao), ao)
                model.initialise_model()
                history = orig_time_series[:pred_idx]
                model.update_model_with_new_data(history)
                mean_pred, lcb, ucb = model.forecast(num_steps_ahead=1, conf_alpha=conf_alpha)
                tot_time = time.time() - start_time
                true_val = orig_time_series[pred_idx]
                err = abs((true_val - mean_pred) / true_val)
                is_in_conf = lcb <= true_val <= ucb
                curr_errs.append(err)
                curr_trapped_by_ci.append(is_in_conf)
                curr_times.append(tot_time)
                curr_conf_widths.append((ucb-lcb)/true_val)
                # print('%s:: true:%0.3f, lcb:%0.3f, est:%0.3f, ucb:%0.3f, trapped:%d, err:%0.3f'%(
                #      str(ao), true_val, lcb, mean_pred, ucb, is_in_conf, err))
                ground_truth.append(true_val)
                esstimate.append(mean_pred)
            errors[ao] = np.mean(curr_errs)
            trapped_by_ci[ao] = np.mean(curr_trapped_by_ci)
            avg_times[ao] = np.mean(curr_times)
            conf_widths[ao] = np.mean(curr_conf_widths)
            print('')
            print('errors', errors)
            print('trapped', trapped_by_ci)
            print('conf_widths', conf_widths)
            print('times', avg_times)

    plt.figure().set_figwidth(20)
    plt.plot(ground_truth, color='g', linewidth=2, label='real load', marker='o')
    plt.plot(esstimate, color='red', linewidth=2, label='load prediction', marker='o')
    # plt.plot(esstimate2, color='pink', linewidth=2, label='load prediction - high bound')
    plt.xlabel("time", fontsize=10)
    plt.legend()

    # plt.savefig("load_demo.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
