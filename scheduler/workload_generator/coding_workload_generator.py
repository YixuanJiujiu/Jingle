from typing import List, Union, Dict

from kubernetes.client import V1Deployment, V1Service, V1ServicePort

from Jingle.scheduler.workload_generator.base_workload_generator import BaseWorkloadGenerator
from Jingle.scheduler.workload_generator.coding_k8s_utils import get_coding_client_template_deployment, \
    get_coding_server_template_deployment


class CodingWorkloadGenerator(BaseWorkloadGenerator):

    def __init__(self, cluster_type=None, Jingle_image=None):
        """ Constructor. """
        self._cluster_type = cluster_type
        if Jingle_image:
            self._Jingle_image = Jingle_image
        elif cluster_type == 'kind':
            self._Jingle_image = "yixuannine/Jinglescheduler:latest"
        else:
            raise ValueError('Invalid input for cluster_type(%s) and/or container_image(%s)' % (
                str(cluster_type), str(Jingle_image)))
        super().__init__()

    def generate_workload_server_objects(self,
                                         app_name: str,
                                         *args,
                                         **kwargs
                                         ) -> List[Union[V1Deployment, V1Service]]:

        # This is the actual workload that must be scaled.
        server_deployment = get_coding_server_template_deployment(app_name,
                                                                  *args,
                                                                  **kwargs)

        return [server_deployment]

    def generate_workload_client_objects(self,
                                         app_name: str,
                                         *args,
                                         Jingle_client_override_args: Dict[str, str] = None,
                                         coding_client_override_args: Dict[str, str] = None,
                                         **kwargs) -> List[Union[V1Deployment, V1Service]]:
        """
        Generates workload client objects. The Jingle client is embedded in the same pod.
        :param Jingle_client_override_args: Args to override for Jingle clients
        :param coding_client_override_args: Args to override for coding clients
        :param app_name: Name of the app in the heirarchy.
        :param kwargs: other kwargs to pass
        :return:
        """
        Jingle_client_override_args = Jingle_client_override_args if Jingle_client_override_args else {}
        coding_client_override_args = coding_client_override_args if coding_client_override_args else {}

        # ========= Generate defaults =============
        DEFAULT_CODING_CLIENT_CMD = ["python", "/driver/wrk_runscript.py"]
        DEFAULT_CODING_CLIENT_ARGS = {
            "--wrk-logdir": "/Jinglelogs",
            "--wrk-qps": "10",
            "--wrk-duration": "30"
        }

        DEFAULT_Jingle_CLIENT_CMD = ["python", "/Jingle/Jingle_clients/drivers/wrk_to_grpc_driver.py"]
        DEFAULT_Jingle_CLIENT_ARGS = {"--log-folder-path": "/Jinglelogs",
                                     "--grpc-port": "10000",
                                     "--grpc-ip": "Jingle-service.default.svc.cluster.local",
                                     "--grpc-client-id": app_name,
                                     "--poll-frequency": "1",
                                     "--slo-type": "latency",
                                     "--slo-latency": 30}

        # =========== Update defaults with workloadinfo args
        coding_args = DEFAULT_CODING_CLIENT_ARGS.copy()
        coding_args.update(coding_client_override_args)

        Jingle_args = DEFAULT_Jingle_CLIENT_ARGS.copy()
        Jingle_args.update(Jingle_client_override_args)

        client_dep = get_coding_client_template_deployment(app_name,
                                                           coding_client_cmd=DEFAULT_Jingle_CLIENT_CMD,
                                                           coding_client_args=coding_args,
                                                           ca_client_cmd=DEFAULT_Jingle_CLIENT_ARGS,
                                                           ca_client_args=Jingle_args,
                                                           *args,
                                                           **kwargs)
        return [client_dep]

    def generate_Jingle_client_objects(self,
                                      app_name: str,
                                      *args,
                                      **kwargs) -> List[Union[V1Deployment, V1Service]]:
        return []  # No Jingle clients for this workload since it is co-located with the workload client
