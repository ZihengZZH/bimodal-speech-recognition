import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


def read_mat(verbose=False):
    # para verbose:
    data = []
    for index in range(1, 23):
        filename = "g%s_aligned.mat" % str(index).zfill(2)
        data.append(loadmat(os.path.join('dataset', 'cuave-group-aligned', filename)))

        if verbose:
            print(filename, "read")

    return data
