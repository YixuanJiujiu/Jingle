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


def get_all():
    methods = ["PRM","PRM+IBT", "PRM+SVR",  "Jingle" ]

    filename_list =[
                    "ds2/workdirs_kind/singlejingle_test1_6_kind_1203210447/root--coding-assign.csv",
                    "ds2/workdirs_kind/binaryjingle_test1_6_kind_1107093645/root--coding-assign.csv",
                    "ds2/workdirs_kind/simplejingle_test1_6_kind_1203230652/root--coding-assign.csv",
                    "ds2/workdirs_kind/jingle_test1_6_kind_1106071810/root--coding-assign.csv",]

    # methods = [30, 60, 90, 120]
    # filename_list = [
    #     "ds2/workdirs_kind/cilantro_test1_36_kind_1208214025_scale4/root--coding-assign.csv",
    #     "ds2/workdirs_kind/cilantro_test1_36_kind_1208224513_scale3/root--coding-assign.csv",
    #     "ds2/workdirs_kind/cilantro_test1_36_kind_1209000717_scale2/root--coding-assign.csv",
    #     "ds2/workdirs_kind/cilantro_test1_6_kind_1106001509/root--coding-assign.csv",
    # ]

    all_load = dict()
    all_alloc = dict()
    all_p99 = dict()
    idx = 0
    length = 122

    #read from file
    for filename in filename_list:
        load = []
        alloc = []
        p99 = []
        list_of_dict = []
        with open(filename, 'r') as f:
            dict_reader = csv.DictReader(f)
            list_of_dict = list(dict_reader)
        for elem in list_of_dict:
            load.append(float(elem['load']))
            p99.append(float(elem['reward']))
            temp = eval(elem['allocs'])
            alloc.append(float(temp['root--coding-assign']))
        # length = min(length, len(alloc))

        all_load[methods[idx]] = load[:length]
        all_alloc[methods[idx]] = alloc[:length]
        all_p99[methods[idx]] = p99[:length]
        idx = idx + 1

    return all_load, all_alloc, all_p99

def plot_single_data(all_data, ylabel):
    time = []
    markers = ['o', 's', '^', 'x']
    for (key, value), marker in zip(all_data.items(), markers):
        time = [i * 30 for i in range(len(value))]
        plt.plot(time, value, label=key, marker = marker)

    if ylabel == "P99 Latency (s)":
        slo = [10 for i in range(len(time))]
        plt.plot(time, slo, linestyle="--")
        plt.text(0, 10, 'SLO (latency)', fontsize=24)
    plt.xlabel("time (s)", fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [4, 0, 1, 2, 3]
    # # order = [0,1]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=22, loc ="upper left")

    # plt.legend(fontsize="18", loc ="upper left")
    plt.legend(fontsize=24)
    plt.legend(fontsize=24, loc='upper left', ncol=4)
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
    markers = ['o', 's', '^', 'x']
    for (key, value), marker in zip(all_load.items(), markers):
        time = [i * 30 for i in range(len(value))]
        plt.plot(time, value, label=key, marker = marker)
        break

    plt.xlabel("time (s)", fontsize=24)
    plt.ylabel("Load\n (number of requests)", fontsize=24)
    # plt.xticks(fontsize=22)
    # plt.yticks(fontsize=
    plt.tight_layout()
    plt.savefig('load.png', dpi=300)
    plt.show()

def plot_learning():
    # Todo
    methods = ["PRM","PRM+IBT", "PRM+SVR",  "Jingle" ]
    mean_allocs = [1.81, 1.79, 1.85, 2.56]
    violations = [50, 18, 46, 1]
    # Creating bar plots
    plt.figure(figsize=(14, 6))
    # MAE plot
    plt.subplot(1, 2, 1)
    plt.bar(methods, mean_allocs, color='skyblue')
    # plt.ylabel('MAE')
    plt.title('Mean Allocations', fontsize=26)
    plt.xticks(methods, rotation=45, fontsize=24)
    plt.yticks(fontsize=24)
    # CSI plot
    plt.subplot(1, 2, 2)
    plt.bar(methods, violations, color='coral')
    # plt.ylabel('CSI')
    plt.title('P99 Violations', fontsize=26)
    plt.xticks(methods, rotation=45, fontsize=24)
    plt.yticks(fontsize=24)

    plt.tight_layout()
    plt.savefig('learning_compare.png', dpi=300)
    plt.show()

def plot_scale():
    scale = [30, 60, 90, 120]

    ca_mean_allocs_jingle = [2.56, 5.19, 12.65, 17.26]
    ca_violations_jingle = [1, 5, 4.6, 0.3]
    ca_violations_jingle = [i/(j*20) for i, j in zip(ca_violations_jingle, scale)]

    ca_mean_allocs_c = [4.36, 16.61, 18.66, 25.5]
    ca_violations_c = [2.67, 0.33, 1.67, 3.667]
    ca_violations_c = [i / (j * 20) for i, j in zip(ca_violations_c, scale)]

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(scale, ca_mean_allocs_jingle, color='blue', marker = 'o', label = 'Jingle')
    plt.plot(scale, ca_mean_allocs_c, color='red', marker = '^', label = 'Cilantro')
    print(ca_violations_c)
    plt.ylim([0, 30])
    plt.title('Mean Allocations at Scale', fontsize=30)
    plt.legend(fontsize=24)
    plt.xticks(scale, fontsize=30)
    plt.yticks(fontsize=30)
    plt.subplot(1, 2, 2)

    plt.plot(scale, ca_violations_jingle, color='blue', marker = 'o', label = 'Jingle')
    plt.plot(scale, ca_violations_c, color='red', marker = '^', label = 'Cilantro')
    plt.title('P99 Violations at Scale', fontsize=30)
    plt.xticks(scale, fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylabel('Percentage of Violations', fontsize=30)
    plt.ylim([0, 0.1])
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig('ca_scale.png', dpi=300)
    plt.show()

    #Jingle data
    ad_mean_allocs_Jingle = [1.88, 3.158, 4.416, 4.915]
    ad_violations_b_jingle = [1, 2, 2, 2]
    ad_violations_b_jingle = [i/(j*20) for i, j in zip(ad_violations_b_jingle, scale)]
    ad_violations_c_jingle = [17, 11, 9.3, 20]
    ad_violations_c_jingle = [i/(j*20) for i, j in zip(ad_violations_c_jingle, scale)]

    # k8s data
    ad_mean_allocs_k8s = [1.88, 3.26, 4.55, 6.033]
    ad_violations_b_k8s = [2, 2, 2, 0.33]
    ad_violations_b_k8s = [i / (j * 20) for i, j in zip(ad_violations_b_k8s, scale)]
    ad_violations_c_k8s = [40, 40, 40, 40]
    ad_violations_c_k8s = [i / (j * 20) for i, j in zip(ad_violations_c_k8s, scale)]

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(scale, ad_mean_allocs_Jingle, color='blue', marker = 'o', label = 'Jingle')
    plt.plot(scale, ad_mean_allocs_k8s, color='red', marker = '<', label = 'K8s')
    plt.title('Mean Allocations at Scale', fontsize=30)
    plt.legend(fontsize=24)
    plt.xticks(scale, fontsize=30)
    plt.yticks(fontsize=30)


    plt.subplot(1, 2, 2)
    plt.plot(scale, ad_violations_b_jingle, color='blue', label = 'Jingle bandwidth', marker = 'o')
    plt.plot(scale, ad_violations_c_jingle, color='blue', label = 'Jingle bandwidth', marker = '^')
    plt.plot(scale, ad_violations_b_k8s, color='red', label = 'K8s bandwidth', marker = '>')
    plt.plot(scale, ad_violations_c_k8s, color='red', label = 'K8s capacity', marker = '<')
    plt.title('Violations at Scale', fontsize=30)
    plt.ylabel('Percentage of Violations', fontsize=30)
    plt.xticks(scale, fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=24)
    plt.ylim([0, 0.1])
    plt.tight_layout()
    plt.savefig('ad_scale.png', dpi=300)
    plt.show()


def plot_all():
    all_load, all_alloc, all_p99 = get_all()

    # for v in all_alloc.values():
    #     print(np.mean(v))
    #
    # # violations
    # testresults = np.vstack((all_p99[30][:120], all_p99[60][:120], all_p99[90][:120], all_p99[120][:120]))
    #
    # count_occurrences = (testresults > 10).astype(int)
    # # Multiply these counts by corresponding elements in array2 and sum up
    # result = np.sum(count_occurrences, axis=1) / 3
    # result.tolist()
    # print("p99 violations" + str(result))
    # return

    # plot_single_load(all_load)
    # plot_single_data(all_load, "Load (number of requests)")
    plt.figure(figsize=(20, 4))
    plt.ylim([0, 8])
    plt.yticks([0, 2, 4, 6])
    plot_single_data(all_alloc, "Allocations")

    plt.figure(figsize=(20, 5))
    plt.ylim([0, 20])
    plt.yticks([0, 5, 10, 15, 20])
    plot_single_data(all_p99, "P99 Latency (s)")


    print("Mean Allocs")
    for key, value in all_alloc.items():
        print(key + " : " + str(mean(value)))

    print("Number of p99 violations")
    for key, value in all_p99.items():
        violations = len([i for i in value if i > 10])
        print(key + " : " + str(violations))


if __name__ == '__main__':
    # plot_all()
    # plot_learning()
    plot_scale()