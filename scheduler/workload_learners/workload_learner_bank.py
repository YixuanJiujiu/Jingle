"""
    A bank for load forecasters.

    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
"""

from Jingle.scheduler.performance_learners.learner_bank import LearnerBank


class TSForecaster:
    """ An abstraction for time series forecasting. """

    def initialise(self):
        """ Initialise. """
        raise NotImplementedError('Implement in a child class.')

    def stop_training_loop(self):
        """ initialise. """
        pass

    def forecast(self, num_steps_ahead, conf_alpha):
        """ Forecasts next element in time series. """
        raise NotImplementedError('Implement in a child class.')


class TSForecasterBank(LearnerBank):
    """ Time series forecaster Bank. """

    @classmethod
    def _check_type(cls, obj):
        """ Checks type. """
        assert isinstance(obj, TSForecaster)

