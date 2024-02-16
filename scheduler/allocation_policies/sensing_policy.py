import logging
from typing import Dict
import time
import collections
import csv
import datetime
import numpy as np

logger = logging.getLogger(__name__)

# local
from Jingle.scheduler.allocation_policies.base_policy import BasePolicy

ROOM_FULL = 35
SENSING_FIEL = '/Jingle/scheduler/allocation_policies/raw_sensing.csv'

# same level with AutoScalingBasePolicy
class OccupancyAdjustedPolicy(BasePolicy):
    def __init__(self, env, resource_quantity, load_forecaster_bank=None,SLO = 10, csv_file_path=SENSING_FIEL):
        super().__init__(env, resource_quantity, load_forecaster_bank, alloc_granularity=1)
        self.occupancy_data = self._read_and_adjust_csv_data(csv_file_path)
        self.SLO = SLO
        self.data_threshold = 3
        self.check_duration = 120 # This is the time-window to detect rapid occupancy changes

    def _read_and_adjust_csv_data(self, csv_file_path):
        occupancy_data = collections.OrderedDict()

        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # skip header

            # Assuming the first row has the earliest timestamp
            first_timestamp = datetime.datetime.strptime(next(reader)[0], "%Y-%m-%d %H:%M:%S.%f")
            time_diff = datetime.datetime.now() - first_timestamp

            # Reset the file reader
            file.seek(0)
            next(reader)  # skip header again

            for row in reader:
                adjusted_timestamp = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f") + time_diff
                occupancy_data[adjusted_timestamp] = int(row[1])

        return occupancy_data

    def _policy_initialise(self):
        # Placeholder for additional initialization
        pass

    def _get_resource_allocation_for_loads(self, curr_loads=None, *args, **kwargs) -> Dict[str, float]:
        raise NotImplementedError('Implement in a child class')

    def _get_autoscaling_resource_allocation_for_loads(self, load, *args, **kwargs):
        """ Get autoscaling resource allocation. """
        raise NotImplementedError('Implement in a child class')

    def get_occupancy_adjustment(self, current_timestamp):
        """Detect rapid changes over the past two minutes in the occupancy data."""
        if not self.occupancy_data:
            return 0
        start_time = current_timestamp - datetime.timedelta(seconds=self.check_duration)
        logger.info(f'*************** occupancy start time: {start_time}; curr-time-stamp: {current_timestamp}; time-delta: {datetime.timedelta(seconds=self.check_duration)}')
        relevant_data = [value for time, value in self.occupancy_data.items() if
                         start_time <= time <= current_timestamp]
        logger.info(
            f'*************** relavant occupancy: {relevant_data};')
        if len(relevant_data) <= self.data_threshold:
            return 0
        increase = relevant_data[0] < relevant_data[-1]
        decrease = relevant_data[0] > relevant_data[-1]
        if increase:
            return 1
        elif decrease:
            return -1
        return 0

class JingleAutoScaler(OccupancyAdjustedPolicy):
    """ optimized with arrival of crowd detection """

    def __init__(self, env, resource_quantity,load_forecaster_bank, performance_recorder_bank,learner_bank,
                 num_recent_event_logs_for_util_computation=20,
                 grid_size_for_util_computation=5, scaling_coeff=1.0, scaling_factor = 0.9):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank)
        self.performance_recorder_bank = performance_recorder_bank
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.scaling_coeff = scaling_coeff # Using 1 seems to cause wide fluctuations
        self.scaling_factor = scaling_factor
        self._last_alloc = None
        self.learner_bank = learner_bank
        self.scaling_down = False

    def _get_resource_allocation_for_loads(self, curr_loads=None, *args, **kwargs) -> Dict[str, float]:
        current_timestamp = datetime.datetime.now()
        adjustment = self.get_occupancy_adjustment(current_timestamp)
        if adjustment > 0:
            curr_loads.update((x, y * 1.7) for x, y in curr_loads.items())
        elif adjustment < 0:
            if self.scaling_down:
                curr_loads.update((x, y * 0.8) for x, y in curr_loads.items())
                self.scaling_down = False
            else:
                self.scaling_down = True
        allocs = self._get_autoscaling_resource_allocation_for_loads(curr_loads, *args, **kwargs)
        tot_alloc = sum([val for _, val in allocs.items()])
        if tot_alloc > self.resource_quantity:
            alloc_ratios = {key: val / tot_alloc for key, val in allocs.items()}
            allocs = self._get_final_allocations_from_ratios(alloc_ratios)
            logger.info(
                'Estimated resource demand is %d, but there are only %d resources.Returning %s.',
                tot_alloc, self.resource_quantity, allocs)
        logger.info(
            "scaling round: " + str(self.round_idx) + "   allocs: " + str(allocs) + "   load predict: " + str(curr_loads))
        return allocs

    def _get_autoscaling_resource_allocation_for_loads(self, load, *args, **kwargs):
        """ Get autoscaling resource allocation. """
        # pylint: disable=unused-argument
        # pylint: disable=broad-except
        SLO = self.SLO
        if self.round_idx <= 2:
            allocs = {key: 1 for key in self.env.leaf_nodes}
        else:
            try:
                recent_allocs_and_util_metrics = \
                    self.performance_recorder_bank.get_recent_allocs_and_util_metrics(
                        num_recent_event_logs=self.num_recent_event_logs_for_util_computation,
                        grid_size=self.grid_size_for_util_computation)
                logger.info("get metrics as: " + str(recent_allocs_and_util_metrics))
                if recent_allocs_and_util_metrics is None:
                    allocs = self._last_alloc
                else:
                    allocs = {}
                    for leaf_path in self.env.leaf_nodes:
                        curr_replicas = recent_allocs_and_util_metrics[0]['alloc'][leaf_path]
                        curr_reward = recent_allocs_and_util_metrics[0]['leaf_rewards'][leaf_path]
                        curr_load = recent_allocs_and_util_metrics[0]['loads'][leaf_path]

                        learner = self.learner_bank.get(leaf_path)
                        option1 = load[leaf_path] * learner.get_recommendation(perf_goal=-SLO * self.scaling_factor,
                                                                               load=1)

                        service_rate = (curr_load * SLO / curr_reward)
                        option2 = int(np.ceil(
                            self.scaling_coeff * load[leaf_path] * curr_replicas / service_rate))
                        option2 = max(1, option2)

                        option = int(np.ceil((option1 + option2) / 2)) if self.round_idx > 20 else int(option2)
                        allocs[leaf_path] = max(1, option)
                        logger.info(f'option1(btree): {option1}; option2 (servicerate): {option2}')
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
        self._last_alloc = allocs
        return allocs



class MJingleAutoScaler(OccupancyAdjustedPolicy):
    """ optimized with arrival of crowd detection for multiple SLO parameters -> throughput & network I/O """

    def __init__(self, env, resource_quantity,load_forecaster_bank, performance_recorder_bank,learner_bank,
                 num_recent_event_logs_for_util_computation=20,
                 grid_size_for_util_computation=5, scaling_coeff=1.0, scaling_factor = 0.9):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank)
        self.performance_recorder_bank = performance_recorder_bank
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.scaling_coeff = scaling_coeff # Using 1 seems to cause wide fluctuations
        self.scaling_factor = scaling_factor
        self._last_alloc = None
        self.learner_bank = learner_bank

    def _get_resource_allocation_for_loads(self, curr_loads=None, *args, **kwargs) -> Dict[str, float]:
        current_timestamp = datetime.datetime.now()
        adjustment = self.get_occupancy_adjustment(current_timestamp)
        if adjustment > 0:
            curr_loads.update((x, y * 1.2) for x, y in curr_loads.items())
        elif adjustment < 0:
            curr_loads.update((x, y * 0.8) for x, y in curr_loads.items())
        allocs = self._get_autoscaling_resource_allocation_for_loads(curr_loads, *args, **kwargs)
        tot_alloc = sum([val for _, val in allocs.items()])
        if tot_alloc > self.resource_quantity:
            alloc_ratios = {key: val / tot_alloc for key, val in allocs.items()}
            allocs = self._get_final_allocations_from_ratios(alloc_ratios)
            logger.info(
                'Estimated resource demand is %d, but there are only %d resources.Returning %s.',
                tot_alloc, self.resource_quantity, allocs)
        logger.info(
            "scaling round: " + str(self.round_idx) + "   allocs: " + str(allocs) + "   load predict: " + str(curr_loads))
        return allocs

    def _get_autoscaling_resource_allocation_for_loads(self, load, *args, **kwargs):
        """ Get autoscaling resource allocation. """
        # pylint: disable=unused-argument
        # pylint: disable=broad-except
        SLO = [4, 1000] # SLO for web serving app - throughput 5, file I/O size 1024 bytes
        if self.round_idx <= 2:
            allocs = {key: 1 for key in self.env.leaf_nodes}
        else:
            try:
                recent_allocs_and_util_metrics = \
                    self.performance_recorder_bank.get_recent_allocs_and_util_metrics(
                        num_recent_event_logs=self.num_recent_event_logs_for_util_computation,
                        grid_size=self.grid_size_for_util_computation)
                logger.info("get metrics as: " + str(recent_allocs_and_util_metrics))
                if recent_allocs_and_util_metrics is None:
                    allocs = self._last_alloc
                else:
                    allocs = {}
                    for leaf_path in self.env.leaf_nodes:
                        curr_replicas = recent_allocs_and_util_metrics[0]['alloc'][leaf_path]
                        # reward is file size per container
                        curr_reward = recent_allocs_and_util_metrics[0]['leaf_rewards'][leaf_path]
                        curr_load = recent_allocs_and_util_metrics[0]['loads'][leaf_path] # load is througput

                        # calculate rate filesize I/O per container
                        service_rate = (curr_load * SLO[1]/curr_reward)
                        option2_1 = int(np.ceil(
                            self.scaling_coeff * load[leaf_path] * curr_replicas / service_rate))
                        logger.info(f'option2(service rate) tp: {service_rate};')
                        # calculate rate tp per container
                        tp = curr_load/curr_replicas
                        service_rate = (curr_load * SLO[0]/tp )
                        logger.info(f'option2(service rate) load: {service_rate};')
                        option2_2 = int(np.ceil(
                            self.scaling_coeff * load[leaf_path] * curr_replicas / service_rate))
                        logger.info(f'option2(service rate details) tp: {option2_1}; load: {option2_2}')
                        option2 = max(1, option2_1, option2_2)

                        # learner is ibtree
                        # learner = self.learner_bank.get(leaf_path)
                        # option1 = load[leaf_path] * learner.get_recommendation(perf_goal=SLO[1],
                        #                                          load=load[leaf_path])

                        # option = int(np.ceil((option1 + option2) / 2)) if self.round_idx > 20 else int(option2)
                        option = int(option2)
                        allocs[leaf_path] = max(1, option)
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
        self._last_alloc = allocs
        return allocs