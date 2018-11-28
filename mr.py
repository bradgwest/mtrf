"""
AUTHOR: Brad West
CREATED ON: 2018-11-25
---
DESCRIPTION: Helper functions for implementing metamorphic relations
"""

import copy
from random import uniform, sample

import numpy as np


def mr_add_uninformative(data, uninformative_value=uniform(-1, 1)):
    """
    Adds a completely uninformative variable to a dataset

    :param data: 2-tuple, the data where the first element is the X
      data and the second is the y (labels)
    :param uninformative_value: int, the value to add to the dataframe
    :return: the data with an added variable
    """
    new_col = np.array([np.array([uninformative_value])
                        for _ in range(len(data[0]))])
    x = np.append(data[0], new_col, 1)  # append column to the end
    return x, data[1]


def compare_results(primary, followup):
    n_diff = 0
    for p, f in zip(primary, followup):
        if p != f:
            n_diff += 1
    return n_diff


def get_transformation_cols(data, n_transform=2):
    """
    Returns a list of columns to transform

    :param data: 2d array, the data to sample from
    :param n_transform: int, the number of columns to transform
    :return:
    """
    if n_transform > len(data[0]):
        old = n_transform
        transformed = len(data[0])
        print("Number of columns to transform is greater than columns "
              "in dataset. Changing parameter from {0} to {1}"
              .format(old, transformed))
    return sample(range(0, n_transform), n_transform)


def mr_linear_transform(data,
                        cols_to_transform,
                        m=uniform(-1, 1),
                        b=uniform(-1, 1)):
    """
    Apply linear transformation of form: mx + b

    For Random forest classifiers, the primary and follow-up test cases
    should remain the same.

    :param data: 2d array, the data to manipulate
    :param cols_to_transform: 1d array, the column indices to transform
    :param m: int, "m" in the equation y = mx + b
    :param b: int, "b" in the equation y = mx + b
    :return: primary and follow-up data
    """

    def linear_eq(x, m, b):
        return m * x + b

    if m == 0:
        m += 0.001
    elif not m:
        m = uniform(-1, 1)

    if not b:
        b = uniform(-1, 1)

    follow_up = copy.deepcopy(data)

    X = follow_up[0]
    for row in X:
        for i in range(0, len(row)):
            if i in cols_to_transform:
                row[i] = linear_eq(row[i], m, b)
    return follow_up
