"""
    A simple event logger - provides no persistence.

    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
"""
import collections

class SimpleEventLogger(object):
    def __init__(self, max_len=1000):
        """
        Constructor.
        """
        self.events = collections.deque(maxlen=max_len)
        super(SimpleEventLogger, self).__init__()

    def log_event(self, event):
        """
        Do nothing, just store event in memory and flush the list if exceeds size.
        """
        self.events.append(event)
