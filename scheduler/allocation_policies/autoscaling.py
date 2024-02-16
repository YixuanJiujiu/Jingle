"""
    Harness for autoscaling from learned utilities.

    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
"""

import logging
import math

import numpy as np
# Local
from Jingle.scheduler.allocation_policies.base_policy import BasePolicy

logger = logging.getLogger(__name__)

MIN_ALLOC_PER_LEAF = 1

class AutoScalingBasePolicy(BasePolicy):
    """ Base class for autoscaling. """

    def __init__(self, env, resource_quantity, load_forecaster_bank=None, trigger_enabled = False, SLO = 10):
        """ Constructor. """
        # Setting alloc_granularity to 1 by default -----------------------------
        super().__init__(env, resource_quantity, load_forecaster_bank, alloc_granularity=1)
        self.trigger_enabled = trigger_enabled
        self.SLO = SLO


    def _policy_initialise(self):
        """ Initialisation in a child class. """
        pass

    def _get_autoscaling_resource_allocation_for_loads(self, load, *args, **kwargs):
        """ Get autoscaling resource allocation. """
        raise NotImplementedError('Implement in a child class')

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Obtain resource allocation for loads. """
        # pylint: disable=arguments-differ
        allocs = self._get_autoscaling_resource_allocation_for_loads(loads, *args, **kwargs)
        tot_alloc = sum([val for _, val in allocs.items()])
        logger.info(f'new allocation from scheduler: {tot_alloc}')
        if tot_alloc > self.resource_quantity:
            alloc_ratios = {key: val / tot_alloc for key, val in allocs.items()}
            allocs = self._get_final_allocations_from_ratios(alloc_ratios)
            logger.info(
                'Estimated resource demand is %d, but there are only %d resources.Returning %s.',
                tot_alloc, self.resource_quantity, allocs)
        return allocs


class SLOAutoScaler(AutoScalingBasePolicy):
    """ Autoscaling based on SLOs. """

    def __init__(self, env, resource_quantity, load_forecaster_bank=None, performance_recorder_bank=None):
        """ Constructor. """
        # Setting alloc_granularity to 1 by default -----------------------------
        super().__init__(env, resource_quantity, load_forecaster_bank)
        self.performance_recorder_bank = performance_recorder_bank
        self.round_idx = 0

    def _get_autoscaling_resource_allocation_for_loads(self, loads, ud_est_type='ucb',
                                                       *args, **kwargs):
        """ Obtain resource allocation for loads. """
        # pylint: disable=arguments-differ
        unit_demands = self.get_unit_demand_estimates(ud_est_type, loads)
        allocs = {}
        for leaf_path, ud in unit_demands.items():
            # get hint from trigger
            allocs[leaf_path] = max(int(np.ceil(ud * loads[leaf_path])), 1)
        self.round_idx = self.round_idx + 1
        logger.info("scaling round: " + str(self.round_idx) + "   allocs: " + str(allocs) + "   unitDemand: " + str(ud_est_type) + "   load predict: " + str(loads))
        return allocs

    def get_unit_demand_estimates(self, ud_est_type, loads):
        raise NotImplementedError('Implement in a child class.')


class BanditAutoScaler(SLOAutoScaler):
    """ Original Bandit auto scaler. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, learner_bank):
        """ Constructor. """
        # Setting alloc_granularity to 1 by default -----------------------------
        super().__init__(env, resource_quantity, load_forecaster_bank)
        self.learner_bank = learner_bank

    def get_unit_demand_estimates(self, est_type, loads):
        """ Obtain unit demand est type. """
        ret_probs = [0.1, 0.9, 1.0]
        ret = {}
        for leaf_path, leaf_node in self.env.leaf_nodes.items():
            learner = self.learner_bank.get(leaf_path)
            ret[leaf_path] = learner.get_recommendation(leaf_node.threshold, 1)

            # chooser = np.random.random()
            # if chooser <= ret_probs[0]:
            #     ret[leaf_path] = learner.get_recommendation_for_lower_bound(leaf_node.threshold, 1)
            # elif chooser <= ret_probs[1]:
            #     ret[leaf_path] = learner.get_recommendation(leaf_node.threshold, 1)
            # else:
            #     ret[leaf_path] = learner.get_recommendation_for_upper_bound(leaf_node.threshold, 1)

        #             if est_type == 'est':
        #                 ret[leaf_path] = learner.get_recommendation(leaf_node.threshold, 1)
        #             elif est_type == 'ucb':
        #                 ret[leaf_path] = learner.get_recommendation_for_upper_bound(leaf_node.threshold,
        #                                                                             1)
        #             elif est_type == 'lcb':
        #                 ret[leaf_path] = learner.get_recommendation_for_lower_bound(leaf_node.threshold,
        #                                                                             1)
        #             else:
        #                 raise ValueError('Unknown est_type %s'%(est_type))
        return ret


class MLAutoScaler(AutoScalingBasePolicy):
    """ Autoscaling based on ML models. """

    def __init__(self, env, resource_quantity, learner_bank, load_forecaster_bank, trigger_enabled=True,
                 num_iters_for_evo_opt=1000):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank, trigger_enabled)
        self.learner_bank = learner_bank
        self.num_iters_for_evo_opt = num_iters_for_evo_opt
        self.trigger_enabled = trigger_enabled

    def _get_autoscaling_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Obtains the allocation. """
        allocs = {}
        if self.round_idx <= 36:
            # varying the allocation at the begining stage -------------------------------------------------
            allocs = {key: (self.round_idx%6 +1) for key in self.env.leaf_nodes}
        else:
            for leaf_path, leaf_node in self.env.leaf_nodes.items():
                learner = self.learner_bank.get(leaf_path)
                allocs[leaf_path] = learner.get_recommendation(loads[leaf_path], )
        return allocs