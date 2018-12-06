import copy
import logging
from random import sample, uniform
import unittest

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier

from make_data import make_data
import mr

N_SAMPLES = [100, 1000, 10000]
N_CLASSES = [(3, 1), (5, 1), (7, 1)]
N_FEATURES = [12]
N_INFO = [(0, 0, 0), (3, 1, 0), (4, 3, 1), (6, 4, 2)]
N_PER = 2

# Classifier Parameters
CLASSIFIER_PARAMS = dict(
    n_estimators=100,
    criterion="gini",
    min_samples_leaf=3,
    max_depth=15,
    max_features="auto",
    oob_score=True,
    random_state=23
)

# Columns of output dataframe
DF_COLS = ["n_samples", "n_features", "n_informative", "n_redundant",
          "n_repeated", "n_classes", "n_clusters_per_class", "seed",
           "mr", "n_diff"]

all_data = make_data(
    N_SAMPLES,
    N_CLASSES,
    N_FEATURES,
    N_INFO,
    N_PER,
    use_seed=True
)

def run(idx):

    print("--Running Job for Index {}--".format(idx))

    # Generate Data
    data = all_data[idx]["data"]
    train_x, train_y = data["train"]
    test_x, test_y = data["test"]

    # Create, fit, and predict a RFC
    initial_clf = RandomForestClassifier(
        n_estimators=CLASSIFIER_PARAMS["n_estimators"],
        criterion=CLASSIFIER_PARAMS["criterion"],
        min_samples_leaf=CLASSIFIER_PARAMS["min_samples_leaf"],
        max_depth=CLASSIFIER_PARAMS["max_depth"],
        max_features=CLASSIFIER_PARAMS["max_features"],
        oob_score=CLASSIFIER_PARAMS["oob_score"],
        random_state=CLASSIFIER_PARAMS["random_state"]
    )
    initial_clf.fit(X=train_x, y=train_y)
    initial_predictions = initial_clf.predict(test_x)

    # Apply linear transform, manipulating the first three columns
    # params = {"cols_to_transform": [0,1,2], "m":-0.5, "b": 1}
    # train_x_2, train_y_2 = mr.mr_linear_transform((train_x, train_y), params)
    # test_x_2, test_y_2 = mr.mr_linear_transform((test_x, train_y), params)

    # params = {"uninformative_value": 0}
    # train_x_2, train_y_2 = mr.mr_add_uninformative((train_x, train_y), params)
    # test_x_2, test_y_2 = mr.mr_add_uninformative((test_x, test_y), params)

    # params = {"new_order": [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]}
    # train_x_2, train_y_2 = mr.mr_reorder_predictors((train_x, train_y), params)
    # test_x_2, test_y_2 = mr.mr_reorder_predictors((test_x, test_y), params)

    train_x_2, train_y_2 = mr.mr_double_dataset((train_x, train_y))
    test_x_2, test_y_2 = mr.mr_double_dataset((test_x, test_y))


    # Create, fit, and predict a follow-up RFC
    follow_up_clf = RandomForestClassifier(
        n_estimators=CLASSIFIER_PARAMS["n_estimators"],
        criterion=CLASSIFIER_PARAMS["criterion"],
        min_samples_leaf=CLASSIFIER_PARAMS["min_samples_leaf"],
        max_depth=CLASSIFIER_PARAMS["max_depth"],
        max_features=CLASSIFIER_PARAMS["max_features"],
        oob_score=CLASSIFIER_PARAMS["oob_score"],
        random_state=CLASSIFIER_PARAMS["random_state"]
    )
    follow_up_clf.fit(X=train_x_2, y=train_y_2)
    follow_up_predictions = follow_up_clf.predict(test_x_2)

    # Print out the results
    def print_results(initial, follow_up):
        num = 0
        for i, f in zip(initial, follow_up):
            if i == f:
                print("({}/{}): Equal".format(num, len(initial)))
            else:
                print("({}/{}): Initial: {}; Follow_up: {}"
                      .format(num, len(initial), i, f))
            num += 1

    def print_results2(initial, follow_up):
        equal = True
        for i, f in zip(initial, follow_up):
            if i != f:
                equal = False
                break
        if not equal:
            print("Did not match")

    print_results2(initial_predictions, follow_up_predictions)
    # print_results(initial_predictions, follow_up_predictions)


for i in range(len(all_data)):
    run(i)
