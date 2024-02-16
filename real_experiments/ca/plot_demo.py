"""
    Plotting tools for the microservices experiment.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
# Local
from numpy import mean

def avg_bar(test_results):
    means = np.mean(test_results, axis=1)
    variances = np.var(test_results, axis=1)
    systems = ['K8s', 'PID', 'Pseudo-DS2', 'Cilantro', 'Jingle']  # The systems we are comparing
    plt.bar(systems, means, yerr=np.sqrt(variances), capsize=6, color='skyblue', alpha=0.7)

def avg_box_bar(test_results):
    systems = ['K8s', 'PID', 'Pseudo-DS2', 'Cilantro', 'Jingle']
    boxplots = plt.boxplot(test_results, labels=systems, patch_artist=True)
    for patch in boxplots['boxes']:
        patch.set_facecolor('skyblue')
    
def avg_timeseries(data, method_name):
    mean_values = np.mean(data, axis=0)
    variance_values = np.var(data, axis=0)
    time = [i * 30 for i in range(len(mean_values))]
    plt.plot(time, mean_values, label=method_name, marker = 'o', markersize = 4)
    plt.fill_between(range(len(mean_values)), mean_values - np.sqrt(variance_values), mean_values + np.sqrt(variance_values), alpha=0.5)
    plt.legend(fontsize=18, loc='best', ncol=5)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()


def get_all():

    files = {}
    filename_list_k8s =[
                    "ds2/workdirs_kind/k8sas_test1_6_kind_1105025516/root--coding-assign.csv",
                    "ds2/workdirs_kind/k8sas_test1_6_kind_1105025516/root--coding-assign.csv",
                    "ds2/workdirs_kind/k8sas_test1_6_kind_1105025516/root--coding-assign.csv",]

    filename_list_pid =[
                    "ds2/workdirs_kind/pidas_test1_6_kind_1105040345/root--coding-assign.csv",
                    "ds2/workdirs_kind/pidas_test1_6_kind_1105040345/root--coding-assign.csv",
                    "ds2/workdirs_kind/pidas_test1_6_kind_1105040345/root--coding-assign.csv",]

    filename_list_ds2 =[
                    "ds2/workdirs_kind/ds2_test1_6_kind_1105195512/root--coding-assign.csv",
                    "ds2/workdirs_kind/ds2_test1_6_kind_1105195512/root--coding-assign.csv",
                    "ds2/workdirs_kind/ds2_test1_6_kind_1105195512/root--coding-assign.csv",]

    filename_list_cilantro =[
                    "ds2/workdirs_kind/cilantro_test1_6_kind_1106001509/root--coding-assign.csv",
                    "ds2/workdirs_kind/cilantro_test1_6_kind_1106001509/root--coding-assign.csv",
                    "ds2/workdirs_kind/cilantro_test1_6_kind_1106001509/root--coding-assign.csv",]

    filename_list_jingle =[
                    "ds2/workdirs_kind/jingle_test1_6_kind_1106071810/root--coding-assign.csv",
                    "ds2/workdirs_kind/jingle_test1_6_kind_1106071810/root--coding-assign.csv",
                    "ds2/workdirs_kind/jingle_test1_6_kind_1106071810/root--coding-assign.csv",]
    # filename_list_k8s = ['ds2/workdirs_kind/jingle_test1_36_kind_1206202341_scale4/root--coding-assign.csv']
    # filename_list_pid = ['ds2/workdirs_kind/jingle_test1_36_kind_1207070150_scale3/root--coding-assign.csv']
    # filename_list_ds2 = ['ds2/workdirs_kind/jingle_test1_36_kind_1207084513_scale2/root--coding-assign.csv']
    files['K8s'] = filename_list_k8s
    files['PID'] = filename_list_pid
    files['Pseudo-DS2'] = filename_list_ds2
    files['Cilantro'] = filename_list_cilantro
    files['Jingle'] = filename_list_jingle

    all_load = dict()
    all_alloc = dict()
    all_p99 = dict()
    length = 120

    #read from file
    for method, filenames in files.items():
        for filename in filenames:
            with open(filename, 'r') as f:
                dict_reader = csv.DictReader(f)
                list_of_dict = list(dict_reader)
            load = []
            alloc = []
            p99 = []
            for elem in list_of_dict:
                load.append(float(elem['load']))
                p99.append(float(elem['reward']))
                temp = eval(elem['allocs'])
                alloc.append(float(temp['root--coding-assign']))
            if not method in all_load.keys():
                all_load[method] = np.array(load[:length])
                all_alloc[method] = np.array(alloc[:length])
                all_p99[method] = np.array(p99[:length])
            else:
                all_load[method] = np.vstack((all_load[method], load[:length]))
                all_alloc[method] = np.vstack((all_alloc[method], alloc[:length]))
                all_p99[method] = np.vstack((all_p99[method], p99[:length]))

    return all_load, all_alloc, all_p99

def plot_single_data(all_data, ylabel):
    plt.figure().set_figwidth(20)
    time = []
    for key, value in all_data.items():
        time = [i * 30 for i in range(len(value))]
        plt.plot(time, value, label=key, marker = 'o')

    if ylabel == "P99 Latency (s)":
        slo = [10 for i in range(len(time))]
        plt.plot(time, slo, label="SLO (Latency)", linestyle="--")
        plt.text(0, 5, 'SLO (latency)', fontsize=24)
    plt.xlabel("time (s)", fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0, 1, 2, 3]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=22, loc ="upper left")
    plt.legend(fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(ylabel + '.png', dpi=300)
    plt.show()
    # plt.xticks(np.arange(min(time), max(time) + 1, 5000))
    # plt.yticks(np.arange(min(alloc), max(alloc) + 1, 1))

def plot_single_load(all_load):
    plt.figure().set_figwidth(20)
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    for key, value in all_load.items():
        time = [i * 30 for i in range(len(value))]
        plt.plot(time, value, label=key, marker = 'o')
        break

    plt.xlabel("time (s)", fontsize=24)
    plt.ylabel("Load\n (number of requests)", fontsize=24)
    # plt.xticks(fontsize=22)
    # plt.yticks(fontsize=
    plt.tight_layout()
    plt.savefig('load.png', dpi=300)
    plt.show()

def plot_all():
    all_load, all_alloc, all_p99 = get_all()

    # plot load
    # plt.figure().set_figwidth(20)
    # avg_timeseries(all_load['K8s'], 'Load\n (number of requests)')
    # plt.xlabel("time (s)", fontsize=24)
    # plt.ylabel("Load\n (number of requests)", fontsize=24)
    # plt.tight_layout()
    # plt.savefig('load.png', dpi=300)
    # plt.show()

    # plot alloc
    plt.figure(figsize=(20, 4))
    for k,v in all_alloc.items():
        avg_timeseries(v, k)
    plt.xlabel("time (s)", fontsize=24)
    plt.ylabel("Allocations", fontsize=24)
    plt.ylim([0, 8])
    plt.yticks([0, 2, 4, 6])
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0, 1, 2, 3]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=22, loc ="upper left")
    # plt.legend(fontsize="18", loc ="upper left")
    plt.tight_layout()
    plt.savefig('ca_allocs.png', dpi=300)
    plt.show()

    # plot latenc
    plt.figure(figsize=(12, 6))
    for k, v in all_p99.items():
        avg_timeseries(v, k)
    time = [i * 30 for i in range(120)]
    slo = [10 for i in range(len(time))]
    plt.plot(time, slo, label="SLO (Latency)", linestyle="--")
    plt.text(0, 5, 'SLO (latency)', fontsize=24)
    plt.xlabel("time (s)", fontsize=24)
    plt.ylabel("P99 Latency (s)", fontsize=24)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0, 1, 2, 3]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=22, loc ="upper left")
    plt.tight_layout()
    plt.savefig('ca_latency.png', dpi=300)
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    methods = ["K8s", "PID", "Pseudo-DS2", "Cilantro", 'Jingle']
    total_violations = [19, 42, 23, 8.7, 8]
    bar2 = plt.bar(range(len(total_violations)), total_violations, width=0.5, tick_label = methods, color='skyblue')
    plt.bar_label(bar2, fontsize=24)
    plt.ylabel("p99 Latency Violations", fontsize=24)
    plt.xticks(fontsize=24, rotation=45)
    plt.yticks(fontsize=24)
    plt.ylim(0, 50)
    plt.tight_layout()

    # mean allocs
    plt.subplot(1, 2, 2)
    testresults = np.vstack((all_alloc['K8s'].flatten(), all_alloc['PID'].flatten(), all_alloc['Pseudo-DS2'].flatten(), all_alloc['Cilantro'].flatten(), all_alloc['Jingle'].flatten()))
    # print(np.mean(testresults, axis=1).reshape(5, 1))
    # avg_bar(testresults)
    # plt.subplot(2, 1, 2)
    avg_box_bar(testresults.T)
    plt.ylabel("Allocations", fontsize=24)
    plt.xticks(fontsize=24, rotation=45)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig('ca_avg.png', dpi=300)
    plt.show()

    #  p99
    # testresults = np.vstack((all_p99['K8s'].flatten(), all_p99['PID'].flatten(), all_p99['Pseudo-DS2'].flatten(), all_p99['Cilantro'].flatten(), all_p99['Jingle'].flatten()))
    # avg_bar(testresults)
    # plt.ylabel("Average P99 Latency \n", fontsize=24)
    # plt.axhline(y=10, color='r', linestyle='--')
    # plt.tight_layout()
    # plt.savefig('avg_latency.png', dpi=300)
    # plt.show()

    # p99 violations
    # mean p99
    # testresults = np.vstack((np.mean(all_p99['K8s'], axis=0), np.mean(all_p99['PID'], axis=0), np.mean(all_p99['Pseudo-DS2'], axis=0), np.mean(all_p99['Cilantro'], axis=0), np.mean(all_p99['Jingle'], axis=0) ))

    testresults = np.vstack((all_p99['K8s'].flatten(), all_p99['PID'].flatten(), all_p99['Pseudo-DS2'].flatten(), all_p99['Cilantro'].flatten(), all_p99['Jingle'].flatten() ))
    count_occurrences = (testresults > 10).astype(int)
    # count_exceed = (testresults - 10)
    # Multiply these counts by corresponding elements in array2 and sum up
    result = np.sum(count_occurrences, axis=1)/3
    # result = np.sum(count_exceed * count_occurrences, axis=1)/3
    result.tolist()
    print(result)

def plot_mean():

    methods = ["K8s", "PID", "Pseudo-DS2", "Cilantro", 'Jingle']

    total_violations = [19, 42, 23, 8.7, 8]

    plt.figure(figsize=(12, 6))
    bar2 = plt.bar(range(len(total_violations)), total_violations, width=0.5, tick_label = methods, color='skyblue')
    plt.bar_label(bar2, fontsize=24)
    plt.ylabel("Total p99 Latency Violations", fontsize=24)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=24)
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.savefig('ca_avg_latency.png', dpi=300)
    plt.show()

def plot_arima():
    methods = [10, 20, 50, 100]

    mae = [1.537, 1.297, 1.323, 1.322]

    plt.figure(figsize=(6, 6))

    # Plot for MAE
    plt.plot(methods, mae, label='MAE', marker='o', color='coral', linewidth = 4)
    plt.ylim([0, 1.6])
    plt.xlabel('Window Size',fontsize=22)
    # plt.ylabel('MAE', fontsize=24)
    plt.title('MAE',fontsize=26)
    plt.xticks(methods, fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig('arima_window.png', dpi=300)
    plt.show()

def plot_workload_predictor():
    # methods = ['ARIMA-IoT', 'ARIMA', 'ARIMA-static','RNN', 'MovingAvg']
    # mae_values = [1.6815, 1.2967, 1.5495, 5.6210, 0.8880]
    # csi_values = [0.444, 0, 0.333, 0, 0.111]
    methods = [ 'MovingAvg','ARIMA', 'RNN', 'ARIMA-static', 'ARIMA-IoT']
    mae_values = [0.8880, 1.2967, 5.621, 1.5495, 1.6815]
    csi_values = [0.111, 0, 0, 0.333, 0.444]
    # Creating bar plots
    plt.figure(figsize=(14, 6))
    # MAE plot
    plt.subplot(1, 2, 1)
    plt.bar(methods, mae_values, color='skyblue')
    # plt.ylabel('MAE')
    plt.title('Mean Absolute Error (MAE)', fontsize=26)
    plt.xticks(methods, rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    # CSI plot
    plt.subplot(1, 2, 2)
    plt.bar(methods, csi_values, color='coral')
    # plt.ylabel('CSI')
    plt.title('Critical Success Index (CSI)', fontsize=26)
    plt.xticks(methods, rotation=45, fontsize=24)
    plt.yticks(fontsize=24)

    plt.tight_layout()
    plt.savefig('wp_csi.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    plot_all()
    # plot_mean()
    # plot_arima()
    # plot_workload_predictor()