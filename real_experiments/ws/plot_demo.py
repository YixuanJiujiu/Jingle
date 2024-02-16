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
    print(means)
    variances = np.var(test_results, axis=1)
    systems = ['K8s', 'Jingle', 'PID']  # The systems we are comparing
    plt.bar(systems, means, yerr=np.sqrt(variances), capsize=6, color='skyblue', alpha=0.7)


def avg_timeseries(data, method_name):
    mean_values = np.mean(data, axis=0)
    variance_values = np.var(data, axis=0)
    time = [i * 30 for i in range(len(mean_values))]
    # print(variance_values)
    plt.plot(time, mean_values, label=method_name, marker = 'o', markersize = 4)
    plt.fill_between(range(len(mean_values)), mean_values - np.sqrt(variance_values), mean_values + np.sqrt(variance_values), alpha=0.5)



def get_all():

    files = {}
    filename_list_k8s =[
                    "ds2/workdirs_kind/k8sas_test2_6_kind_1120041243/root--assign-distribute.csv",
                    "ds2/workdirs_kind/k8sas_test2_6_kind_1120041243/root--assign-distribute.csv",
                    "ds2/workdirs_kind/k8sas_test2_6_kind_1120041243/root--assign-distribute.csv",]
    # filename_list_k8s = ["ds2/workdirs_kind/multijingle_test2_6_kind_1207025127_scale4/root--assign-distribute.csv"]
    # filename_list_jingle = ["ds2/workdirs_kind/multijingle_test2_6_kind_1207041536_scale3/root--assign-distribute.csv"]
    # filename_list_pid = ["ds2/workdirs_kind/multijingle_test2_6_kind_1207052919_scale2/root--assign-distribute.csv"]
    filename_list_jingle =[
                    "ds2/workdirs_kind/multijingle_test2_6_kind_1120181817/root--assign-distribute.csv",
                    "ds2/workdirs_kind/multijingle_test2_6_kind_1120181817/root--assign-distribute.csv",
                    "ds2/workdirs_kind/multijingle_test2_6_kind_1120181817/root--assign-distribute.csv",]

    filename_list_pid = [
        "ds2/workdirs_kind/pid_test2_6_kind_1201055734/root--assign-distribute.csv",
        "ds2/workdirs_kind/pid_test2_6_kind_1201055734/root--assign-distribute.csv",
        "ds2/workdirs_kind/pid_test2_6_kind_1201055734/root--assign-distribute.csv", ]



    files['K8s'] = filename_list_k8s
    files['Jingle'] = filename_list_jingle
    files['pid'] = filename_list_pid

    # files[30] = ['ds2/workdirs_kind/k8sas_test2_36_kind_1209012253/root--assign-distribute.csv']
    # files[60] = ['ds2/workdirs_kind/k8sas_test2_36_kind_1209023526/root--assign-distribute.csv']
    # files[90] = ['ds2/workdirs_kind/k8sas_test2_36_kind_1209041559/root--assign-distribute.csv']
    # files[120] = ['ds2/workdirs_kind/k8sas_test2_6_kind_1120155857/root--assign-distribute.csv']
    all_load = dict()
    all_alloc = dict()
    all_p99 = dict()
    all_cap = dict()
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
            cap = []
            for elem in list_of_dict:
                load.append(float(elem['load']))
                p99.append(float(elem['reward']))
                temp = eval(elem['allocs'])
                alloc.append(float(temp['root--assign-distribute']))
                cap.append(float(elem['load'])/float(temp['root--assign-distribute']))
            if not method in all_load.keys():
                all_load[method] = np.array(load[:length])
                all_alloc[method] = np.array(alloc[:length])
                all_p99[method] = np.array(p99[:length])
                all_cap[method] = np.array(cap[:length])
            else:
                all_load[method] = np.vstack((all_load[method], load[:length]))
                all_alloc[method] = np.vstack((all_alloc[method], alloc[:length]))
                all_p99[method] = np.vstack((all_p99[method], p99[:length]))
                all_cap[method] = np.vstack((all_cap[method], cap[:length]))


    return all_load, all_alloc, all_p99, all_cap

def plot_single_data(all_data, ylabel):
    plt.figure().set_figwidth(20)
    time = []
    for key, value in all_data.items():
        time = [i * 30 for i in range(len(value))]
        plt.plot(time, value, label=key, marker = 'o')

    if ylabel == "P99 Latency (s)":
        slo = [10 for i in range(len(time))]
        plt.plot(time, slo, label="SLO (Latency)", linestyle="--")
        # plt.text(0, 5, 'SLO (latency)', fontsize=24)
    plt.xlabel("time (s)", fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0, 1, 2, 3]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=22, loc ="upper left")

    plt.legend(fontsize="18")
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
    plt.savefig('ad_load.png', dpi=300)
    plt.show()

def plot_all():
    all_load, all_alloc, all_p99, all_caps = get_all()
    
    # for v in all_alloc.values():
    #     print(np.mean(v))
    #
    # # violations
    # testresults = np.vstack((all_p99[30][:120], all_p99[60][:120], all_p99[90][:120], all_p99[120][:120]))
    # capresults = np.vstack((all_p99[30][:120], all_p99[60][:120], all_p99[90][:120], all_p99[120][:120]))

    # count_occurrences = (testresults > 1000).astype(int)
    # # Multiply these counts by corresponding elements in array2 and sum up
    # result = np.sum(count_occurrences, axis=1)/3
    # result.tolist()
    # print( "bandwidth violations"+ str(result))
    #
    # count_occurrences = (capresults > 4).astype(int)
    # # Multiply these counts by corresponding elements in array2 and sum up
    # result = np.sum(count_occurrences, axis=1) / 3
    # result.tolist()
    # print("capacity violations" + str(result))
    # return

    # plot load
    # plt.figure().set_figwidth(20)
    # avg_timeseries(all_load['K8s'], 'Load\n (number of requests)')
    # plt.xlabel("time (s)", fontsize=24)
    # plt.ylabel("Load\n (number of requests)", fontsize=24)
    # plt.tight_layout()
    # plt.savefig('ad_load.png', dpi=300)
    # plt.show()

    # plot alloc
    plt.figure(figsize=(20, 4))
    for k, v in all_alloc.items():
        avg_timeseries(v, k)
    plt.xlabel("time (s)", fontsize=24)
    plt.ylabel("Allocations", fontsize=24)
    plt.ylim([0, 8])
    plt.yticks([0, 2, 4, 6])
    plt.legend(fontsize=24, loc='upper left', ncol=3)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig('ad_allocs.png', dpi=300)
    plt.show()
    return

    # plot bandwidth
    plt.figure().set_figwidth(20)
    for k, v in all_p99.items():
        avg_timeseries(v, k)
    time = [i * 30 for i in range(120)]
    slo = [1000 for i in range(len(time))]
    plt.plot(time, slo, label="SLO (Latency)", linestyle="--")
    # plt.text(0, 5, 'SLO (bandwidth)', fontsize=24)
    plt.xlabel("time (s)", fontsize=24)
    plt.ylabel("Bandwidth Per Container", fontsize=24)
    plt.legend(fontsize="18", loc ="upper left")
    plt.tight_layout()
    plt.savefig('ad_bandwidth.png', dpi=300)
    plt.show()

    # plot capacity
    plt.figure().set_figwidth(20)
    for k, v in all_caps.items():
        avg_timeseries(v, k)
    time = [i * 30 for i in range(120)]
    slo = [4 for i in range(len(time))]
    plt.plot(time, slo, label="SLO (Capacity)", linestyle="--")
    # plt.text(0, 5, 'SLO (Capacity)', fontsize=24)
    plt.xlabel("time (s)", fontsize=24)
    plt.ylabel("Capacity Per Container", fontsize=24)
    plt.legend(fontsize="18", loc ="upper left")
    plt.tight_layout()
    plt.savefig('ad_capacity.png', dpi=300)
    plt.show()

    # mean allocs
    testresults = np.vstack((all_alloc['K8s'].flatten(), all_alloc['Jingle'].flatten(), all_alloc['pid'].flatten()))
    avg_bar(testresults)
    plt.ylabel("Average Allocation \n", fontsize=24)
    plt.tight_layout()
    plt.savefig('ad_avg_alloc.png', dpi=300)
    plt.show()

    # violations
    testresults = np.vstack((all_p99['K8s'].flatten(), all_p99['Jingle'].flatten(), all_p99['pid'].flatten()))
    capresults = np.vstack((all_caps['K8s'].flatten(), all_caps['Jingle'].flatten(), all_caps['pid'].flatten()))

    count_occurrences = (testresults > 1000).astype(int)
    # Multiply these counts by corresponding elements in array2 and sum up
    result = np.sum(count_occurrences, axis=1)/3
    result.tolist()
    print( "bandwidth violations"+ str(result))

    count_occurrences = (capresults > 4).astype(int)
    # Multiply these counts by corresponding elements in array2 and sum up
    result = np.sum(count_occurrences, axis=1) / 3
    result.tolist()
    print("capacity violations" + str(result))


def plot_bar():
    # Algorithms and their corresponding violations
    algorithms = ['k8s', 'PID', 'Jingle']
    bandwidth_violations = [1, 18, 1]
    capacity_violations = [26, 30, 17]

    # Creating bar plots
    plt.figure(figsize=(14, 6))
    # 1 plot
    plt.subplot(1, 2, 1)
    plt.bar(algorithms, bandwidth_violations, color='skyblue', width=0.5)
    # plt.ylabel('MAE')
    plt.title('Bandwidth Violations', fontsize=30)
    plt.xticks(algorithms, rotation=45, fontsize=30)
    plt.yticks(fontsize=30)
    plt.yticks([0, 5, 10, 15, 20])
    # 2 plot
    plt.subplot(1, 2, 2)
    plt.bar(algorithms, capacity_violations, color='coral', width=0.5)
    # plt.ylabel('CSI')
    plt.title('P99 Violations', fontsize=30)
    plt.xticks(algorithms, rotation=45, fontsize=30)
    plt.yticks(fontsize=30)
    plt.yticks([0, 10, 20, 30])
    plt.tight_layout()
    plt.savefig('ad_violations.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    # plot_all()
    plot_bar()