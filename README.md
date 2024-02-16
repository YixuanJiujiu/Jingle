# Jingle: IoT-Informed Autoscaling for Efficient Resource Management in Edge Computing

Edge computing is increasingly applied to various systems for its proximity to end-users and data sources. To facili-
tate the deployment of diverse edge-native applications, container technology has emerged as a favored solution due to
its simplicity in development and resource management. However, deploying edge applications at scale can quickly
overwhelm edge resources, potentially leading to violations of service-level objectives (SLOs). Scheduling edge
containerized applications to meet SLOs while efficiently managing resources is a significant challenge. In this paper,
we introduce Jingle, an autonomous autoscaler for edge clusters designed to efficiently scale edge-native applications.
Jingle utilizes application performance metrics and domain- specific insights collected from IoT devices to construct a
hybrid model. This hybrid model combines a predictive-reactive module with a lightweight learning model. We demonstrate
Jingle’s effectiveness through a real-world deployment in a classroom setting, managing two edge-native applications
across edge configurations. Our experimental results show that Jingle can fulfill SLO requirements while limiting CPU
usage, requiring up to 50% fewer containers than a state-of-the-art cloud scheduler, highlighting its resource
management efficiency.

## System Requirements

- Hardware:
    - The primary experiments are performed on a personal workstation equipped with an Apple M1 Pro chip, complemented
      by 16 GB of RAM. But this hardware is not critical when reproducing the project.

- Software Requirements:
    - Docker and Kubernetes are required. We also deploy Kind cluster to perform single edge node experiments.

## Usage Guide

This Jingle experiments focus on two edge native applications:

1) An assignment distribution application, powered by Nginx, allowing students to access and download assign- ments,
   where the capacity of concurrent connections and bandwidth of transmitted file size are the SLOs;
2) A coding assignment application, where students submit code and validate it with test cases, with latency as SLO;

The workload generators for the applications are different. Switching workload generator require re-modify the images of
Jingle scheduler, thus we provide the guide to run experiments of coding assignment. You can simply modify the
worker/workload/driver.py to switch the workload generator and re-build the images of Jingle scheduler.

## Data Traces

### Workload Traces

Both applications take HTTP requests as workload requests. We synthesized request traces based on actual classroom
activity data from 2023 Oct. 16th, between 12:56 pm and 1:56 pm, which is a university recitation section with 30
students. We have implemented the workload generator and it is deployed within the deploy.sh file. You can check the
details of workload traces in the [worker/workload/](worker/workload/)

### IoT Sensing Traces

Jingle utilizes an Aqara Motion Sensor P1 and a Raspberry Pi that collect crowd-moving data. To enable reproducibility,
we provide the real-world students-moving data as sensing traces in [scheduler/allocation_policies/raw_sensing.csv](scheduler/allocation_policies/raw_sensing.csv).

### Coding Assignment Application

To reproduce the experiments with coding assignment application, you can use this command to run the tests with all
baselines and Jingle, this test shall take at least 5 hours to finish the script.

```
cd ./real_experiments/ca

./starter/create_kind_cluster.sh # this creates a kind cluster

./ds2/deploy_all.sh
```

In addition, to run a single round for a specific autoscaling algorithm, you need to follow these commands

```
cd ./real_experiments/ca

./starter/create_kind_cluster.sh # this creates a kind cluster

./ds2/deploy.sh # The default deploy script runs Jingle scheduler. To test with other baselines, manually modify the line 9 of deploy.sh to deploy other schedulers
```

During the experiments, you can view the container logs through:

```
./starter/view_logs.sh
```

You shall allow for about one hour to finish the experiment. Then you can fetch the experimental results by:

```
./ds2/fetch_results.sh
```

After fetching results, the experiments results are stored in [./real_experiments/ca/ds2/workdirs_kind/](./real_experiments/ca/ds2/workdirs_kind/).

## Plot Results

We also provide the plotting function in [./real_experiments/ca/plot_demo.py](./real_experiments/ca/plot_demo.py).

To plot the experimental results, you need to modify the folder names in plot_demo.py from line 41-65. Replacing it with
your evaluation results that are stored in the folder [./real_experiments/ca/ds2/workdirs_kind/](./real_experiments/ca/ds2/workdirs_kind/).

After replacing the directory names, you can plot the results by:

```
python plot_demo.py
```