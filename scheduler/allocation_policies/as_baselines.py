"""
    Implements k8s's default autoscaling policy.

    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
"""

import logging
import numpy as np
# Local
from Jingle.scheduler.allocation_policies.autoscaling import AutoScalingBasePolicy

logger = logging.getLogger(__name__)


class K8sAutoScaler(AutoScalingBasePolicy):
    """ K8s autoscaling. """

    def __init__(self, env, resource_quantity, performance_recorder_bank,
                 num_recent_event_logs_for_util_computation=20,
                 grid_size_for_util_computation=5, scaling_coeff=1.0):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank=None)
        self.performance_recorder_bank = performance_recorder_bank
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.scaling_coeff = scaling_coeff # Using 1 seems to cause wide fluctuations
        self._last_alloc = None

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
                logger.info("k8s got info as: " + str(recent_allocs_and_util_metrics))
                if recent_allocs_and_util_metrics is None:
                    allocs = self._last_alloc
                else:
                    allocs = {}
                    for leaf_path, leaf in self.env.leaf_nodes.items():
                        curr_replicas = recent_allocs_and_util_metrics[0]['alloc'][leaf_path]
                        curr_reward = recent_allocs_and_util_metrics[0]['leaf_rewards'][leaf_path]
                        # the k8s default operating on ratio of mmetrics
                        curr_reward = curr_reward/self.SLO

    #                     allocs[leaf_path] = int(curr_replicas * leaf.threshold / curr_reward)
    #                     allocs[leaf_path] = int(np.round(self.scaling_coeff * curr_replicas *
    #                                                      leaf.threshold / curr_reward))
                        allocs[leaf_path] = max(int(np.ceil(self.scaling_coeff * curr_replicas *
                                                        curr_reward)), 1)
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
        self._last_alloc = allocs
        return allocs


class PIDAutoScaler(AutoScalingBasePolicy):
    """ PID controller. """

    def __init__(self, env, resource_quantity, performance_recorder_bank,
                 p_coeff=5, i_coeff=0.000, d_coeff=2,
                 num_recent_event_logs_for_util_computation=20,
                 grid_size_for_util_computation=5):
        """ Controller. """
        super().__init__(env, resource_quantity, load_forecaster_bank=None)
        self.performance_recorder_bank = performance_recorder_bank
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.p_coeff = p_coeff
        self.i_coeff = i_coeff
        self.d_coeff = d_coeff
        self._last_alloc = None
        self._curr_errs = {key: 0 for key in self.env.leaf_nodes}
        self._sum_errs = {key: 0 for key in self.env.leaf_nodes}
        self._diff_errs = {key: 0 for key in self.env.leaf_nodes}

    def _update_error_coeffs(self, recent_allocs_and_util_metrics):
        SLO = self.SLO
        """ Update error parameters. """
        for leaf_path, leaf_node in self.env.leaf_nodes.items():
            curr_err = recent_allocs_and_util_metrics[0]['leaf_rewards'][leaf_path] - SLO
            self._diff_errs[leaf_path] = curr_err - self._curr_errs[leaf_path]
            self._sum_errs[leaf_path] += curr_err
            # Swap current errors ----------------------------------------------------
            self._curr_errs[leaf_path] = curr_err

    def _get_autoscaling_resource_allocation_for_loads(self, load, *args, **kwargs):
        """ Get autoscaling resource allocation. """
        # pylint: disable=unused-argument
        # pylint: disable=broad-except
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
                    self._update_error_coeffs(recent_allocs_and_util_metrics)
                    allocs = {}
                    for leaf_path in self.env.leaf_nodes:
                        curr_replicas = recent_allocs_and_util_metrics[0]['alloc'][leaf_path]
                        change = curr_replicas * (
                            self.p_coeff * self._curr_errs[leaf_path] +
                            self.i_coeff * self._sum_errs[leaf_path] +
                            self.d_coeff * self._diff_errs[leaf_path])
                        allocs[leaf_path] = int(max(1, curr_replicas + change))
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
        self._last_alloc = allocs
        return allocs

class DS2AutoScaler(AutoScalingBasePolicy):
    """ Pseudo-DS2 auto scaler. """

    def __init__(self, env, resource_quantity,load_forecaster_bank, performance_recorder_bank,
                 num_recent_event_logs_for_util_computation=20,
                 grid_size_for_util_computation=5, scaling_coeff=1.0):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank)
        self.performance_recorder_bank = performance_recorder_bank
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.scaling_coeff = scaling_coeff # Using 1 seems to cause wide fluctuations
        self._last_alloc = None

    def _get_autoscaling_resource_allocation_for_loads(self, load, *args, **kwargs):
        """ Get autoscaling resource allocation. """
        # pylint: disable=unused-argument
        # pylint: disable=broad-except
        SLO = self.SLO
        logger.info(f"ds2 round ix as: {self.round_idx}")
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
                        # calculate the true processing rate ---------------------------------------
                        true_processing_rate = curr_load / (curr_replicas * curr_reward)
                        # calculate the expected processing rate ---------------------------------------
                        expected_processing_rate = curr_load / (curr_replicas * self.SLO)
                        # allocs[leaf_path] = max(int(np.round(
                        #     self.scaling_coeff * load[leaf_path] * curr_replicas /
                        #     true_processing_rate)), 1)]
                        diff = np.ceil((expected_processing_rate - true_processing_rate)/true_processing_rate * curr_replicas) + curr_replicas
                        allocs[leaf_path] = max(int(np.round(diff)), 1)
                        logger.info(f'expected ratio: {expected_processing_rate};true ratio: {true_processing_rate}; adjust allocs: {allocs[leaf_path]}')
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
        self._last_alloc = allocs
        return allocs

class SimpleJingleAutoScaler(AutoScalingBasePolicy):
    """ hybrid with svr. Used for multi SLO parameters """

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

                        service_rate = (curr_load * SLO/curr_reward)
                        option2 = int(np.round(
                            self.scaling_coeff * load[leaf_path] * curr_replicas /
                            service_rate))
                        option2 = max(1, option2)

                        learner = self.learner_bank.get(leaf_path)
                        pref_pred = - learner.get_recommendation(alloc_pred=option2,
                                                               load=load[leaf_path])

                        if pref_pred>0 and pref_pred > SLO*self.scaling_factor:
                            allocs[leaf_path] = option2 + 1
                        else:
                            allocs[leaf_path] = int(option2)
                        logger.info(f'pref_pred (sgd): {pref_pred}; option2 (svr): {option2}')
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
        self._last_alloc = allocs
        return allocs


class BinaryJingleAutoScaler(AutoScalingBasePolicy):
    """ hybrid with ibtree. """

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
                        option1 = load[leaf_path] * learner.get_recommendation(perf_goal= -SLO*self.scaling_factor, load=1)

                        service_rate = (curr_load * SLO/curr_reward)
                        option2 = int(np.round(
                            self.scaling_coeff * load[leaf_path] * curr_replicas /service_rate))
                        option2 = max(1, option2)

                        option = int(np.ceil((option1+option2)/2)) if self.round_idx>20 else int(option2)
                        allocs[leaf_path] = max(1, option)
                        logger.info(f'option1(btree): {option1}; option2 (servicerate): {option2}')
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
        self._last_alloc = allocs
        return allocs

class SingleJingleAutoScaler(AutoScalingBasePolicy):
    """ hybrid with ibtree. """

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
                        service_rate = (curr_load * SLO/curr_reward)
                        option2 = int(np.round(
                            self.scaling_coeff * load[leaf_path] * curr_replicas /service_rate))
                        option2 = max(1, option2)
                        allocs[leaf_path] = int(option2)
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
        self._last_alloc = allocs
        return allocs