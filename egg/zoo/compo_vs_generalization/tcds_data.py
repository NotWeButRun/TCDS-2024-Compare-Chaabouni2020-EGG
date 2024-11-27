""" IEEE TCDS 2024 の比較実験に使うためのモジュール """
import itertools

iters = [range(4) for _ in range(4)]
data = list(itertools.product(*iters))
TRAIN_DATA = [
    [0, 0, 0, 0],
    [1, 2, 1, 2],
    [0, 0, 0, 0],
    [1, 2, 1, 2],
    [1, 2, 1, 2],
    [0, 0, 0, 0],
    [1, 2, 1, 2],
]

TEST_DATA = [
    [0, 1, 0, 1],
    [1, 3, 1, 3],
    [0, 0, 0, 0],
    [1, 2, 1, 2],
    [0, 0, 0, 0],
    [1, 2, 1, 2],
]

import torch
import numpy as np

def tidyup_receiver_output(n_attributes, n_values, receiver_output):
    for receiver_output_i in receiver_output:
        receiver_output_i = np.array(receiver_output_i)
        reshaped_output = np.reshape(receiver_output_i, (n_attributes, -1))
        assert reshaped_output.shape == (n_attributes, n_values), reshaped_output.shape
        print(reshaped_output.argmax(axis=1))
