"""
    A bank is to store various objects relevant for each application.
    These objects could be learners, time_series_learners, or performance recorders

    The following code is adapted from
    Bhardwaj, Romil, et al. "Cilantro:{Performance-Aware} resource allocation for general objectives via online feedback."
    17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). USENIX Association, 2023.
    For more information, visit https://github.com/romilbhardwaj/cilantro.
"""


class Bank:
    """ Learner Bank. """

    def __init__(self):
        """ Constructor. """
        super().__init__()
        self._bank = {}

    def tag_exists(self, tag):
        """ Returns true if the tag exists. """
        return tag in self._bank

    @classmethod
    def _check_type(cls, obj):
        """ Checks the type of the object. Can be over-ridden in a child class. """
        # pylint: disable=unused-argument
        return True

    def register(self, tag, obj):
        """ Register the learner. """
        # assert not self.tag_exists(tag)
        self._check_type(obj)
        self._bank[tag] = obj

    def get(self, tag):
        """ Obtain the learner. """
        if self.tag_exists(tag):
            return self._bank[tag]
        else:
            return None

    def get_tags(self):
        """ Returns all tags. """
        return list(self._bank)

    def enumerate(self):
        """ Enumerates learners. """
        return self._bank.items()

    def delete(self, tag):
        """ Delete learner. """
        if self.tag_exists(tag):
            self._bank.pop(tag)

