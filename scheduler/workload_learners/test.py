"""
    A simple demo using time series.

"""

# pylint: disable=too-many-locals

import os
import time
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# Local


def main():
    """ Main function. """
    Allocs = [{"root--coding-assign": 6}]
    Load = [163]
    loc = []
    for alloc in Allocs:
        for k, v in alloc.items():
            loc.append(v)
    X = [temp / load for temp, load in zip(loc, Load)]
    print(X)


if __name__ == '__main__':
    main()
