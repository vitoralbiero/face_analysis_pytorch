from os import path

import bcolz
import numpy as np


def get_val_pair(folder_path, name):
    samples = bcolz.carray(rootdir=path.join(folder_path, name), mode="r")
    issame = np.load(path.join(folder_path, "{}_list.npy".format(name)))

    return samples, issame
