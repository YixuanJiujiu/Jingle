"""
    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
"""


import time
from argparse import ArgumentParser

import pandas

class TimeseriesDataSource():
    def __init__(self,
                 csv_file: str,
                 roll_over: bool = True
                 ):
        '''
        Timeseries datasource that .
        :param csv_file: path to csv file. CSV format is [time, load, reward]
        :param roll_over: CSV loops infinitely if set true
        '''
        self.csv_file = csv_file
        self.roll_over = roll_over

        self.df = pandas.read_csv(csv_file)
        self.max_time = max(self.df['time'])
        self.last_get_time = 0  # Time stamp of last sample get call
        self.start_time = time.time()

    def set_start_time(self):
        self.start_time = time.time()

    def get_mean_val(self, start_time, end_time):
        idxs = self.df['time'].between(start_time, end_time)
        return self.df[idxs].mean()

    def get_data(self):
        now = time.time()
        start_time = self.last_get_time-self.start_time
        end_time = now - self.start_time
        if self.roll_over:
            # Assuming get_samples is called more frequently than max_time, roll over condition is a simple %
            start_time %= self.max_time
            end_time %= self.max_time
        if not self.roll_over and (end_time > self.max_time):
            raise StopIteration('All values have been iterated over.')
        data = self.get_mean_val(start_time=start_time,
                                         end_time=end_time)
        self.last_get_time = now
        if data.isnull().values.all():
            return None
        return dict(data)

    @classmethod
    def add_args_to_parser(self,
                           parser: ArgumentParser):
        parser.add_argument('--csv-file', '-f', type=str, default="",
                            help='Path to CSV file.')
        parser.add_argument('--roll-over', '-r', action="store_true", default=False,
                            help='If specified, rolls over the CSV file to produce infinite stream.')
