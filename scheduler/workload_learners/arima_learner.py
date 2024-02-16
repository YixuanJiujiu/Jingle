""""
    An API for ARIMA time series model.

    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
"""

import logging
import warnings
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
# Local
from Jingle.scheduler.workload_learners.base_load_learner import TSModel


# Ignore convergence warning
logger = logging.getLogger(__name__)
warnings.simplefilter('ignore', ConvergenceWarning)


DFLT_MAX_DATA_LENGTH = 20
MIN_SAMPLES_REQD_FOR_ARIMA = 3
# 2 is the minimum, otherwise problem happens

class ARIMATSModel(TSModel):
    """ ARIMA Time series model. """

    def __init__(self, name, order, max_data_length=DFLT_MAX_DATA_LENGTH):
        """ Constructor. """
        super().__init__(name)
        self.data = []
        self.max_data_length = max_data_length
        self.model = None
        self.model_fit = None
        self.order = order

    # Add/process data -----------------------------------------------------------------------------
    def update_model_with_new_data(self, X):
        """ Updates model with the given data. """
        self.data.extend(X)
        self.data = self.data[-self.max_data_length:]
        if len(self.data) >= MIN_SAMPLES_REQD_FOR_ARIMA:
            self._fit_model()
        else:
            logger.debug('Not training TS model %s since num-data = %d.', self.name, len(self.data))

    def _initialise_model_child(self):
        """ Initialise model. """
        pass

    def _fit_model(self):
        """ Fits model. """
        self.model = ARIMA(self.data, order=self.order)
        self.model_fit = self.model.fit()

    # Forecasting ---------------------------------------------------------------------------------
    def forecast(self, num_steps_ahead=1, conf_alpha=0.2):
        """ Forecast TS model. """
        # pylint: disable=bare-except
        if not self.data:
            return 1, 1, 1
        else:
            try:
                arima_alpha = (1 - conf_alpha)
                forecast = self.model_fit.get_forecast(num_steps_ahead)
                pred_mean = forecast.predicted_mean[-1]
                conf_ints = forecast.conf_int(alpha=arima_alpha)
                lcb = conf_ints[-1][0]
                ucb = conf_ints[-1][1]
                if ucb < 0:
                    ucb = 0
                logger.info('[ARIMA: %s] Forecasted with %d data. prediction is %d',
                             self.name, len(self.data), ucb)
            except:
                pred_mean = np.mean(self.data)
                ret_std = np.std(self.data)
                lcb = pred_mean - 2 * conf_alpha * ret_std
                ucb = pred_mean + 2 * conf_alpha * ret_std
                logger.info('[ARIMA: %s] Unable to forecast with %d data.',
                             self.name, len(self.data))
            return pred_mean, lcb, ucb