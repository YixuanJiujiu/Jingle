#!/usr/bin/env bash
set -e

CONFIG_PATH=$1

# If CONFIG_PATH is not provided, use the default kubeconfig
if [ -z "$CONFIG_PATH" ]; then
    CONFIG_PATH=~/.kube/config
fi

JinglePOD=$(kubectl get pods --kubeconfig ${CONFIG_PATH} | awk '/Jinglescheduler/ {print $1;exit}')
kubectl cp $JinglePOD:/Jingle/workdirs ./workdirs_kind/ --kubeconfig ${CONFIG_PATH}
# Copy experiments logs:
LATESTDIR=$(ls -td ./workdirs_kind/*/ | head -1)
kubectl logs $JinglePOD --kubeconfig ${CONFIG_PATH} > ${LATESTDIR}Jinglescheduler.log
echo Results are fetched in workdirs_kind