"""
AUTHOR Brad West
CREATED ON: 2018-11-25
---
DESCRIPTION: Unit testing for metamorphic relations for testing sklearn
  Random Forest Classifier
"""

import copy
import logging
from random import sample, uniform
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from make_data import make_data
import mr

logging.getLogger().setLevel(logging.INFO)

# Output file
OUT = "../data/out.csv"

# Dataset Parameters
N_SAMPLES = [100, 1000, 10000]
N_CLASSES = [(3, 1), (5, 1), (7, 1)]
N_FEATURES = [12]
N_INFO = [(0, 0, 0), (3, 1, 0), (4, 3, 1), (6, 4, 2)]
N_PER = 2

# Create Data
ALL_DATA = make_data(
    N_SAMPLES,
    N_CLASSES,
    N_FEATURES,
    N_INFO,
    N_PER,
    use_seed=True
)

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

# Create initial dataframe
pd.DataFrame([DF_COLS]).to_csv(OUT, index=False, header=False)


def print_results(initial, follow_up):
    equal = True
    for i, f in zip(initial, follow_up):
        if i != f:
            equal = False
            break
    if not equal:
        print("Did not match")


def run_mr(data, config, metrel, mr=None, **kwargs):

    # results
    results = copy.deepcopy(config)
    results["mr"] = mr

    # Generate Data
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

    # Apply metamorphic relation to data
    train_x_2, train_y_2 = metrel((train_x, train_y), kwargs)
    test_x_2, test_y_2 = metrel((test_x, test_y), kwargs)

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

    num_diff = analyze_results(initial_predictions, follow_up_predictions)
    if num_diff > 0:
        print("Did Not Match. Number different: {} of {}"
              .format(num_diff, len(initial_predictions)))

    results["n_diff"] = num_diff
    return results, num_diff


def analyze_results(initial, follow_up):
    num_diff = 0
    for i, f in zip(initial, follow_up):
        if i != f:
            num_diff += 1
    return num_diff


def running_job_log(mr, idx):
    print("--MR {}: Running Job for Index {}--".format(mr, idx))


def write_results(results):
    df = pd.DataFrame(results, columns=DF_COLS, dtype=np.int64)
    with open(OUT, "a") as f:
        df.to_csv(f, index=False, header=False)
    logging.info("Wrote data to {}".format(OUT))


class RFCTester(unittest.TestCase):

    def test_mr1(self):
        """
        MR 1 -- Linear transformation
        """
        all_results = []
        for i in range(len(ALL_DATA)):
            running_job_log(1, i)
            m = uniform(-1, 1)
            b = uniform(-1, 1)
            cols_to_transform = sample(range(N_FEATURES[0]), 7)
            results, num_diff = run_mr(ALL_DATA[i]["data"],
                                       ALL_DATA[i]["config"],
                                       mr.mr_linear_transform,
                                       1,
                                       cols_to_transform=cols_to_transform,
                                       m=m,
                                       b=b)
            """
            For some reason multiplying the feature by a negative number seems
            to cause it to not match
            """
            all_results.append(results)
            with self.subTest(i=i):
                self.assertEqual(num_diff, 0)

        write_results(all_results)

    def test_mr2(self):
        """
        MR 2 -- Addition of uninformative variable
        """
        all_results = []
        for i in range(len(ALL_DATA)):
            running_job_log(1, i)
            uninformative_value = 0
            results, num_diff = run_mr(data=ALL_DATA[i]["data"],
                                       config=ALL_DATA[i]["config"],
                                       metrel=mr.mr_add_uninformative,
                                       mr=2,
                                       uninformative_value=uninformative_value)
            all_results.append(results)
            with self.subTest(i=i):
                self.assertEqual(num_diff, 0)

        write_results(all_results)

    def test_mr3(self):
        """
        MR 3 -- Re-order predictors
        """
        all_results = []
        for i in range(len(ALL_DATA)):
            running_job_log(1, i)
            new_order=[11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
            results, num_diff = run_mr(data=ALL_DATA[i]["data"],
                                       config=ALL_DATA[i]["config"],
                                       metrel=mr.mr_reorder_predictors,
                                       mr=3,
                                       new_order=new_order)
            all_results.append(results)
            with self.subTest(i=i):
                self.assertEqual(num_diff, 0)

        write_results(all_results)

    def test_mr4(self):
        """
        MR 4 -- Increase size of dataset
        """
        all_results = []
        for i in range(len(ALL_DATA)):
            running_job_log(1, i)
            results, num_diff = run_mr(data=ALL_DATA[i]["data"],
                                       config=ALL_DATA[i]["config"],
                                       metrel=mr.mr_double_dataset,
                                       mr=4)
            all_results.append(results)
            with self.subTest(i=i):
                self.assertEqual(num_diff, 0)

        write_results(all_results)
