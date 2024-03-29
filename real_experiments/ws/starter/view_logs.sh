#!/usr/bin/env bash
set -e

CONFIG_PATH=$1

# If CONFIG_PATH is not provided, use the default kubeconfig
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=~/.kube/config
fi

kubectl logs -f $(kubectl get pods --kubeconfig ${CONFIG_PATH} | awk '/Jinglescheduler/ {print $1;exit}') --kubeconfig ${CONFIG_PATH}
