"""
Publishes data through grpc.

    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
"""

import logging
import os
from argparse import ArgumentParser
from typing import Dict

import grpc

from Jingle.utility.grpc.protogen import utility_update_pb2_grpc, utility_update_pb2

CLIENT_RETCODE_SUCCESS = 1
CLIENT_RETCODE_FAIL = 2

logger = logging.getLogger(__name__)


class GRPCPublisher():
    def __init__(self,
                 client_id: str,
                 ip: str = None,
                 port: int = None,
                 timeout: float = 1):
        """
        Sends the data over GRPC as a UtilityMessage.
        :param client_id: Used as the publish tag for grpc messages.  If None, reads envvars set by Kubernetes.
        :param ip: IP of GRPC server to publish samples at. If None, reads envvars set by Kubernetes.
        :param port: GRPC server port
        :param timeout: Timeout after which the publish will be failed.
        """
        self.client_id = client_id
        if ip is None:
            # If ip is not specified, assume running inside k8s cluster and find the service
            ip = os.environ['Jingle_SERVICE_SERVICE_HOST']
        if port is None:
            port = os.environ['Jingle_SERVICE_SERVICE_PORT']
        self.ip = ip
        self.port = port
        self.timeout = timeout

    def publish(self, data: Dict) -> [int, str]:
        """
        Publishes data to the output grpc stub and returns a ret code.
        :param data: Dictionary of data to be published
        :return: retcode, 1 if successful, 2 if fail. Also returns an error string
        """
        with grpc.insecure_channel(f'{self.ip}:{self.port}') as channel:
            stub = utility_update_pb2_grpc.UtilityMessagingStub(channel)
            load = float(data['load'])
            alloc = float(data['alloc'])
            reward = float(data['reward'])
            sigma = float(data['sigma'])
            event_start_time = float(data['event_start_time'])
            event_end_time = float(data['event_end_time'])
            debug = str(data['debug']) if 'debug' in data.keys() else None
            # debug='True'
            msg = utility_update_pb2.UtilityMessage(app_id=self.client_id,
                                                    load=load,
                                                    alloc=alloc,
                                                    reward=reward,
                                                    sigma=sigma,
                                                    event_start_time=event_start_time,
                                                    event_end_time=event_end_time,
                                                    debug=debug)
            logger.debug(f"Publishing msg: {msg}")
            try:
                stub.PublishUtility(msg,
                                    timeout=self.timeout)
                ret = CLIENT_RETCODE_SUCCESS, None
            except grpc._channel._InactiveRpcError as e:
                if (e._state.code == grpc.StatusCode.DEADLINE_EXCEEDED) or (
                        e._state.code == grpc.StatusCode.UNAVAILABLE):
                    ret = CLIENT_RETCODE_FAIL, e._state.details
                    logger.info(f"Publish failed: {e._state.details}")
                else:
                    raise e
            return ret

    @classmethod
    def add_args_to_parser(self,
                           parser: ArgumentParser):
        parser.add_argument('--grpc-port', '-p', type=int, default=None, help='GRPC Port')
        parser.add_argument('--grpc-ip', '-i', type=str, default=None, help='GRPC IP address')
        parser.add_argument('--grpc-client-id', '-c', type=str, default="TSClient",
                            help='Name to be used for publishing utility messages.')
