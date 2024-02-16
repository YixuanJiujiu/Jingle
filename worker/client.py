'''
    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
'''
import argparse
import logging
import time
from typing import Union, List

from Jingle.worker.alloc_source import K8sAllocSource
from Jingle.worker.k8s_data_source import AllocationBasedDataSource
from Jingle.worker.grpc_publisher import GRPCPublisher, CLIENT_RETCODE_SUCCESS

logger = logging.getLogger(__name__)


class Client(object):
    def __init__(self,
                 data_source: AllocationBasedDataSource,
                 publisher: Union[GRPCPublisher, List[GRPCPublisher]],
                 poll_frequency: float,
                 alloc_source: K8sAllocSource = None,
                 ):
        '''
        Base worker that polls a data_source and sends the data to the publisher.
        :param data_source: DataSource object to poll for data
        :param publisher: Publisher object to send data to or a list of publisher objects.
        :param poll_frequency: How frequently to poll data from data_source
        :param alloc_source: Allocation source to use to fetch current allocation if 'alloc' is not returned by data source.
        '''
        self.data_source = data_source
        if not isinstance(publisher, list):
            publisher = [publisher]
        self.publishers = publisher
        self.poll_frequency = poll_frequency
        self.alloc_source = alloc_source

    def run_loop(self):
        while True:
            try:
                data = self.data_source.get_data()
            except StopIteration as e:
                logger.info(f"{str(e)}")
                break
            if data is not None:
                # print(f"Publishing data to grpc server: {data}")
                if 'alloc' not in data and self.alloc_source is not None:
                    data['alloc'] = self.alloc_source.get_allocation()
                for p in self.publishers:
                    ret, msg = p.publish(data)
                    if ret != CLIENT_RETCODE_SUCCESS:
                        logger.warning(f"Publishing to {type(p)} failed, not retrying. Error: {msg}")
            time.sleep(self.poll_frequency)

    @classmethod
    def add_args_to_parser(self,
                           parser: argparse.ArgumentParser):
        parser.add_argument('--poll-frequency', '-pf', type=float, default=1,
                            help='How frequently to poll the data source.')
