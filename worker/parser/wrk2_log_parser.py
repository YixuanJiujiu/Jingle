"""
    wrk2 output parser.
    Parses the output from wrk2, the http load generator used by microservices
    applications (such as hotel reservation).

    To produce example output: ./wrk -R 100 -D exp -t 16 -c 16 -d 30 -L -s \
    ./scripts/hotel-reservation/mixed-workload_type_1.lua \
    http://frontend.default.svc.cluster.local:5000 > output.txt

    First line is the vector of resource allocations
    Second line is the target qps
    event_start_time: and event_end_time: are added in the file by client
    Following lines are the output from wrk

    This client embeds a lot of data in the debug str as json. Example:
    {
      "runtime": 29.98,
      "throughput": 2976.15,
      "num_operations": 89213,
      "avg_latency": 38.749,
      "stddev_latency": 48.459,
      "p50": 18.29,
      "p90": 112.51,
      "p99": 190.08,
      "p999": 244.74,
      "p9999": 294.4,
      "p100": 331.52,
      "event_start_time": 1649991801.0285008,
      "event_end_time": 1649991831.0650449,
      "target_qps": 3000,
      "load": 89940,
      "allocs": {
        "root--consul": 8,
        "root--frontend": 8,
        "root--memcached-profile": 8,
        "root--memcached-rate": 8,
        "root--memcached-reserve": 8,
        "root--mongodb-profile": 8,
        "root--mongodb-rate": 8,
        "root--mongodb-recommendation": 8,
        "root--mongodb-reservation": 8,
        "root--profile": 8,
        "root--search": 8
      }
    }
"""

import json
from argparse import ArgumentParser
import logging
import re
from typing import Dict

from Jingle.worker.parser.base_log_parser import BaseLogParser

logger = logging.getLogger(__name__)


class WrkLogParser(BaseLogParser):
    """ Parser for wrk. """

    def __init__(self):
        """ Constructor. """
        super().__init__()

    def get_data(self, log_file) -> Dict:
        """
        This method returns a dict with data by parsing the log_file
        This method also computes the utilities from the raw metrics.
        :return:
        """
        # pylint: disable=not-callable
        data = {}
        with open(log_file, 'r') as f:
            log_data = f.read()
        # Extract first line
        lines = log_data.split('\n')
        allocs = json.loads(lines[0])
        load = int(lines[1])
        reward = float(lines[2])
        data['event_start_time'] = float(lines[3].split(':')[1])
        data['event_end_time'] = float(lines[4].split(':')[1])
        data['load'] = load
        data['allocs'] = allocs
        data['reward'] = data['p99'] = reward
        # Return -----------------------------------------------------------------------------
        print(data)
        print_str = ""
        for key, value in data.items():
            if isinstance(value, float):
                # Format the float to have 3 decimal places
                value_str = "{:.3f}".format(value)
                print_str += f"{key}: {value_str}; "
            else:
                print_str += f"{key}: {value}; "

        ret = {'load': data['load'],
               'reward': data['reward'],
               'alloc': -1,  # Allocs are sent in debug_str
               'sigma': -1,
               'event_start_time': data['event_start_time'],
               'event_end_time': data['event_end_time'],
               'debug': json.dumps(data)}
        return ret

    @classmethod
    def add_args_to_parser(cls,
                           parser: ArgumentParser):
        """
        Adds class specific arugments to the given argparser.
        Useful to quickly add args to driver script for different sources.
        :param parser: argparse object
        :return: None
        """
        pass

