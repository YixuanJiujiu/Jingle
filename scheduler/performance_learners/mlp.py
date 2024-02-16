"""
    A MLP Learning model. used for single-component application
"""

from argparse import Namespace
import logging
from sklearn.neural_network import MLPRegressor
import numpy as np
# local
from Jingle.scheduler.performance_learners.base_learner import LearningModel


logger = logging.getLogger(__name__)

LOAD_NORMALISER = 100


class MLP(LearningModel):
    """ A learning model. """
    # pylint: disable=abstract-method

    def __init__(self, name, descr, alloc_leaf_order, options=None):
        """ Constructor.
            - alloc_leaf_order is the order in which allocations for each leaf will be stored in the
                               model.
            - app_client_key is the key to search for in the Rewards and Loads dictionaries for the
                               application's reward and leaf.
        """
        super().__init__(name, int_lb=None, int_ub=None)
        self.alloc_leaf_order = alloc_leaf_order
        self.num_dims = len(alloc_leaf_order)
        self.descr = descr
        self.all_inputs = []
        self.all_labels = []
        self.total_data = 0
        self.curr_model = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)

    def _initialise_model_child(self):
        """ Initialises model. """
        pass

    def update_model_with_new_data(self, Allocs, Rewards, Loads, Sigmas, Event_times):
        """ Updates the model with new data. """
        logger.info('--------------Update MLP model---------------')
        num_new_data = 0
        new_input = None
        new_label = None
        for alloc, reward, load in zip(Allocs, Rewards, Loads):
            logger.info('data: ' + str(Allocs) )
            logger.info('alloc: ' + str(alloc) + ',   reward: ' + str(reward) + ',   load: ' + str(load))
            # new_input = [reward, load]
            new_input = np.reshape([reward, load], (1, -1))
            new_label = [sum([alloc[leaf] for leaf in self.alloc_leaf_order])]
            self.all_inputs.append(new_input)
            self.all_labels.append(new_label)
            num_new_data += 1
            logger.info('data: ' + str(new_input) + ',   label: ' + str(new_label))
        if num_new_data == 0: # Return if there is no new data
            return
        self.total_data += num_new_data
        logger.info(f'Updated model with {num_new_data} data')
        self.curr_model.partial_fit(new_input, new_label)

    def get_mean_pred_and_std_for_alloc_load(self, load):
        """ Returns the prediction for perf_goal and load. """
        dat_in = np.reshape([0, load], (1, -1))
        pred = self.curr_model.predict(dat_in)
        logger.info('dat_in: %s', str(dat_in))
        logger.info('pred: %s', pred)
        return max(1, int(pred[0]))

