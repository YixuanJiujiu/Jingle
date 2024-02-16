"""
    A linear regression Learning model. used for single-component application
"""

from argparse import Namespace
import logging
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# local
from Jingle.scheduler.performance_learners.base_learner import LearningModel


logger = logging.getLogger(__name__)

# Hyperparameters
GAMMA = 0.1  # Smaller gamma means larger similarity radius in RBFSampler
ALPHA = 0.001  # Higher alpha means stronger regularization
LR = 'adaptive'  # Adaptive learning rate
ETA0 = 0.1  # Initial learning rate

LOSS_THRESHOLD = 1

class SGDLearner(LearningModel):
    """ A learning model. """
    # pylint: disable=abstract-method

    def __init__(self, name, descr, alloc_leaf_order, n_attributes=2):
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
        self.rbf_feature = RBFSampler(gamma=GAMMA, random_state=1, n_components=n_attributes)
        self.curr_model = SGDRegressor(max_iter=1, penalty='l2', alpha=ALPHA, learning_rate=LR, eta0=ETA0)
        self.scaler = StandardScaler()
        self._current_loss = float('inf')
        self._current_r2 = 0

    def _initialise_model_child(self):
        """ Initialises model. """
        pass

    def update_model_with_new_data(self, Allocs, Rewards, Loads, Sigmas, Event_times):
        """ Updates the model with new data. """
        num_new_data = 0
        new_input = None
        new_label = None
        for alloc, reward, load in zip(Allocs, Rewards, Loads):
            # new_input = [reward, load]
            allocs = sum([alloc[leaf] for leaf in self.alloc_leaf_order])
            new_input = np.reshape([allocs, load], (1, -1))
            # new_label = [sum([alloc[leaf] for leaf in self.alloc_leaf_order])]
            new_label = [reward]
            self.all_inputs.append(new_input)
            self.all_labels.append(new_label)
            num_new_data += 1
        if num_new_data == 0: # Return if there is no new data
            return
        self.total_data += num_new_data
        X_scaled = self.scaler.fit_transform(new_input)
        self.rbf_feature.fit(X_scaled)
        X_features = self.rbf_feature.transform(X_scaled)
        self.curr_model.partial_fit(X_features, new_label)
        #update the loss:
        y_pred = self.curr_model.predict(X_features)
        self._current_loss = mean_squared_error(new_label, y_pred)
        logger.info(f'**************** SGD training ** input: {X_features} {new_label}; predtions: {y_pred}; loss: {self._current_loss} *********************')

    def get_mean_pred_and_std_for_alloc_load(self, alloc_pred, load, *args, **kwargs):
        """ Returns the prediction for perf_goal and load. """
        dat_in = np.reshape([alloc_pred, load], (1, -1))
        X_scaled = self.scaler.fit_transform(dat_in)
        X_features = self.rbf_feature.transform(X_scaled)
        pred = self.curr_model.predict(X_features)
        logger.info(f'**************** SGD prediction: {pred} input: {dat_in}; scaled: {X_features} *********************')
        return pred if self._current_loss <= LOSS_THRESHOLD else 0

