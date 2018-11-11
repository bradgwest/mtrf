"""
AUTHOR: Brad West
CREATED ON: 2018-11-10
---
DESCRIPTION: Functionality for simulating data for testing random forests with
  metamorphic relations.
"""

import numpy as np
from sklearn.datasets import make_classification
import random


N_SAMPLES = [100, 1000, 10000]
N_CLASSES = [(3, 1), (5, 1), (7, 1)]
N_FEATURES = [12]
N_INFO = [(0, 0, 0), (3, 1, 0), (4, 3, 1), (6, 4, 2)]
N_PER = 2


def generate_data(n_samples,
                  n_classes,
                  n_features,
                  n_info,
                  n_per,
                  use_seed=True):
    """
    Generate data for testing metamorphic relations

    Generates random data for a range of parameters. Each combination of the
    various parameters will be used

    :param n_samples: list, the number of examples to use
    :param n_classes: list of 2-tuples. First tuple element is the number of
      classes, second element is the number of clusters per class.
    :param n_features: list, the number of predictors to generate
    :param n_info: list of 2-tuples. First element in a tuple is number of
      informative predictors, second element is the number of repeated predictors.
      Third element is the number of redundant predictors.
    :param n_per: int, number of repeated datasets at each combination.
    :param use_seed: Boolean, should random seeds be used for the generation?
      Each dataset will get it's own seed
    :return: list of dictionaries, with data and metadata about the parameters
      used to construct the dataset.
    """
    # Total number of datasets
    tot = len(n_samples) * len(n_classes) * len(n_features) * len(n_info) * n_per

    if use_seed:
        seeds = [i for i in range(1, tot + 1)]
    else:
        seeds = random.sample(range(1, tot*1000), tot)

    dsets = [dict(data=None, config=dict()) for _ in range(0, tot)]
    j = 0

    for x in n_samples:
        for k, c in n_classes:
            for m in n_features:
                for i, r, s in n_info:
                    for rep in range(0, n_per):
                        d = dsets[j]
                        # make_classification cannot generate uninformative data with multiple classes
                        if i == 0:
                            d["data"] = make_uninformative_classifier(n_samples=x,
                                                                      n_features=m,
                                                                      n_classes=k,
                                                                      random_state=seeds[j])
                        else:
                            d["data"] = make_classification(n_samples=x,
                                                            n_features=m,
                                                            n_informative=i,
                                                            n_redundant=r,
                                                            n_repeated=s,
                                                            n_classes=k,
                                                            n_clusters_per_class=c,
                                                            random_state=seeds[j])
                        d["config"]["n_samples"] = x
                        d["config"]["n_features"] = m
                        d["config"]["n_informative"] = i
                        d["config"]["n_redundant"] = r
                        d["config"]["n_repeated"] = s
                        d["config"]["n_classes"] = k
                        d["config"]["n_clusters_per_class"] = c
                        d["config"]["seed"] = j
                        j += 1
    return dsets


def make_uninformative_classifier(n_samples, n_features, n_classes, random_state=23):
    """
    Generate uninformative data

    :param n_samples: int, the number of samples
    :param n_features: int, the number of features, all of which will be uninformative
      and drawn independently from a N(0,1) distribution.
    :param n_classes: int, the number of classes to assign
    :param random_state: the seed for the random number generator
    :return: A 2-tuple. First element is a two-dimensional array, n_samples * n_features,
      and second element is 1d array of length n_samples. Output mimics that of
      sklearn.datasets.make_classification
    """
    # Set Seeds
    np.random.seed(random_state)
    random.seed(random_state)
    # Generate data
    X = [np.random.normal(size=n_samples) for _ in range(0, n_features)]
    y = [random.randrange(0, n_classes) for _ in range(0, n_samples)]
    return X, y
