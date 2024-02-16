import json
import csv
import logging
import os
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import requests
from typing import Dict, List
from kubernetes import client, config

logger = logging.getLogger(__name__)



class WebDriver(object):
    K8S_NAMESPACE = 'default'
    MICROSERVICES = ['coding-assign']
    # MICROSERVICES = ['assign-distribute']
    # REQUEST_DATASET = '/Jingle/worker/workload/scale4_request.csv'
    # REQUEST_DATASET = '/Jingle/worker/workload/scale3_request.csv'
    REQUEST_DATASET = '/Jingle/worker/workload/scale2_request.csv'

    def __init__(self, logdir: str = None, url: str = 'http://frontend.default.svc.cluster.local:5000', ms: str = MICROSERVICES[0]):
        self.log_output_dir = logdir
        self.wrk_url = url
        self.ms = ms

        # Init k8s clients
        self.load_k8s_config()
        self.appsapi = client.AppsV1Api()
        self.resource_allocs = self.get_alloc()
        logger.info(f"Got an initial resource allocs: {self.resource_allocs}")

        # Start thread to update resource count in background.
        resource_update_thread = threading.Thread(
            target=self.update_resource_alloc_thread, args=())
        resource_update_thread.start()

    def load_k8s_config(self):
        if os.getenv('KUBERNETES_SERVICE_HOST'):
            logger.debug('Detected running inside cluster. Using incluster auth.')
            config.load_incluster_config()
        else:
            logger.debug('Using kube auth.')
            config.load_kube_config()

    def update_resource_alloc_thread(self, sleep_time=1):
        while True:
            new_resource_alloc = self.get_alloc()
            if new_resource_alloc != self.resource_allocs:
                logger.info(f"Got new resource count! Setting resources to {new_resource_alloc}")
                self.resource_allocs = new_resource_alloc
            time.sleep(sleep_time)

    def get_alloc(self) -> Dict[str, int]:
        try:
            deps = self.appsapi.list_namespaced_deployment(namespace=self.K8S_NAMESPACE)
            deps = {d.metadata.name: d for d in deps.items if
                    d.metadata.name.replace('root--', '') in self.MICROSERVICES}
            current_allocations = {dep_name: d.status.ready_replicas for dep_name, d in deps.items()}
            return current_allocations
        except Exception as e:
            logger.error("Failed to get deployment list.")
            return {}

    def send_request(self, data, headers, timeout_seconds=30):
        start_time = time.time()
        try:
            start_time = time.time()
            if self.ms == 'coding-assign':
                response = requests.post(self.wrk_url, data=data, headers=headers, timeout=timeout_seconds)
                end_time = time.time()
                latency = end_time - start_time  # Calculate the latency
                return start_time, latency
            else:
                response = requests.get(self.wrk_url, timeout=timeout_seconds)
                return start_time, len(response.content)

        except requests.Timeout:
            logger.error("Request timed out after {} seconds".format(timeout_seconds))
            return start_time, timeout_seconds

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    def send_concurrent_requests(self, count, data, headers):
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [executor.submit(self.send_request, self.wrk_url, data, headers) for _ in range(count)]
            for future in futures:
                status, content = future.result()
                if status != 200:
                    logger.warning(f"Received non-200 response: {status}, {content}")

    def write_output_to_disk(self,
                             avg_allocs: Dict[str, float],
                             load: int,
                             reward: float,
                             event_start_time: float,
                             event_end_time: float,
                             wrk_stdout: str
                             ):
        """
        Writes the utility message to a log file on disk.
        The utility message is written as a json.
        :return:
        """
        timestr = time.strftime("%Y%m%d-%H%M%S-%f")[:-3]
        log_filename = "output_%s.log" % timestr  # This will be the final name of the log
        log_filepath = os.path.join(self.log_output_dir, log_filename)

        with open(log_filepath, 'w') as f:
            f.write(json.dumps(avg_allocs) + '\n')
            f.write(str(load) + '\n')
            f.write(str(reward) + '\n')
            f.write(f"event_start_time:{event_start_time}" + '\n')
            f.write(f"event_end_time:{event_end_time}" + '\n')
            f.write(wrk_stdout)

    @staticmethod
    def average_list_of_dictionaries(list_of_dicts: List[Dict[str, int]]) -> Dict[str, float]:
        """
        Given a list of dictionaries, returns a dictionary of the average values for each key.
        :param list_of_dicts: List of dictionaries.
        :return: Dictionary of average values for each key.
        """
        avg_dict = {}
        for key in list_of_dicts[0].keys():
            avg_dict[key] = sum([d[key] for d in list_of_dicts if d[key] is not None]) / len(list_of_dicts)
        return avg_dict

    def _read_dataset(self, filename: str) -> List[int]:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            return [int(row[1]) for row in reader]

    @staticmethod
    def p99_latency(latencies: List[float]):
        sorted_latencies = sorted(latencies)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[index]

    def run_loop(self):

        dataset = self._read_dataset(self.REQUEST_DATASET)
        data = "code=#include <iostream>%0A  using namespace std; %0Aint main (){%0Aint n,s,ans,a[4],i;%0A while(cin>>n){	%0As=0;%0A		while(n--){	%0Aans=0;%0A 	for(i=0;i<3;i%2B%2B) {%0A	cin>>a[i];%0A	if (a[i]==1)%0A	ans%2B%2B; } %0A	if (ans>=2)	s%2B%2B;  %0A } cout<<s<<endl;  } return 0;}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        while True:
            # Event to signal the start of a batch
            batch_event = threading.Event()

            # Timer thread to signal the event every 30 seconds
            def timer_thread():
                while True:
                    time.sleep(30)
                    batch_event.set()

            # Start the timer thread
            threading.Thread(target=timer_thread, daemon=True).start()

            for load in dataset:
                # Wait for the batch event
                batch_event.wait()
                batch_event.clear()

                # Send concurrent requests based on the load
                start_times = []
                allocs = []
                latencies = []

                with ThreadPoolExecutor(max_workers=load) as executor:
                    results = list(executor.map(lambda _: self.send_request(data, headers), range(load)))

                allocs.append(self.resource_allocs)
                print(f' ------------------ list allocs: {allocs}')
                # Gather statistics (like latencies and resource allocations) from 'results' if needed
                for result in results:
                    if result:
                        start_times.append(result[0])
                        latencies.append(result[1])

                if not start_times:
                    continue

                # Getting the start and end time for this batch of requests
                event_start_time = min(start_times)
                event_end_time = max([st + lt for st, lt in zip(start_times, latencies)])
                # Average resource allocations
                avg_allocs = self.average_list_of_dictionaries(allocs)
                if self.ms == 'coding-assign':
                    reward = self.p99_latency(latencies)
                else:
                    # for web serving application
                    reward = sum(latencies)/ (sum(avg_allocs.values())/len(avg_allocs))
                print(f' ------------------ client allocs: {avg_allocs}')
                # Write results to disk
                self.write_output_to_disk(avg_allocs,
                                          load,
                                          reward,
                                          event_start_time,
                                          event_end_time,
                                          "\n".join(map(str,
                                                        latencies)))  # Converting latencies to string as a placeholder for stdout
