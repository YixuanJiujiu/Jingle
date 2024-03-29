"""
* This was adapted from https://www.usenix.org/conference/osdi23/presentation/bhardwaj
"""

'''
    The following code is adapted from 
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback." 
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro. 
'''
import logging
from argparse import ArgumentParser
from kubernetes import client

from Jingle.utility.kube_manager import load_k8s_config, LABEL_IS_WORKLOAD

logger = logging.getLogger(__name__)

class K8sAllocSource(object):
    """
    Kubernetes allocation source - fetches the allocation from kubernetes API .
    """
    def __init__(self,
                 app_name: str,
                 namespace: str = 'default'):
        """
        :param app_name: Name of the kubernetes application to fetch allocation for
        :param namespace: Kubernetes namespace to operate in
        """
        self.app_name = app_name
        self.namespace = namespace
        load_k8s_config()
        self.coreapi = client.CoreV1Api()
        self.appsapi = client.AppsV1Api()

    def get_allocation(self) -> int:
        deps = self.appsapi.list_namespaced_deployment(namespace=self.namespace)
        deps = {d.metadata.name: d for d in deps.items if d.metadata.labels.get(LABEL_IS_WORKLOAD, "false").lower() == "true"}  # Filter only deployments which are actual workloads. This is to exclude cilantro worker and app worker deployments.

        # Use ready replicas instead of total replicas to get actual current resource allocation
        current_allocations = {d.metadata.name: d.status.ready_replicas for dep_name, d in deps.items()}
        try:
            alloc = int(current_allocations[self.app_name])
        except KeyError as e:
            logger.error(f"ERROR: Key {self.app_name} not found in current allocs {current_allocations}.")
            raise e
        return alloc


    @classmethod
    def add_args_to_parser(self,
                           parser: ArgumentParser):
        parser.add_argument('--app-name', '-apn', type=str,
                            help='App name (used for fetching and reporting allocations)')