"""
AUTHOR Brad West
CREATED ON: 2018-11-25
---
DESCRIPTION: Unit testing for metamorphic relations for testing sklearn
  Random Forest Classifier
"""

from make_data import make_data
import mr
import unittest
from sklearn.ensemble import RandomForestClassifier
from random import uniform
import copy
import pandas as pd
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)

# Output file
OUT = "../data/out.csv"

# Dataset Parameters
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


class RFCTester(unittest.TestCase):

    def setUp(self):
        """
        Generate data for all tests

        All primary data is the same. Follow-up data is generated in
        each test (an individual MR). Metadata represents the
        configuration for the dataset
        """
        # Generate data
        self.all_data = make_data(
            N_SAMPLES,
            N_CLASSES,
            N_FEATURES,
            N_INFO,
            N_PER,
            use_seed=True
        )
        self.n = len(self.all_data)
        logging.info("Finished generating datasets. {} Generated"
                     .format(self.n))
        self.primary_classifier = [None for _ in range(self.n)]
        self.metadata = [None for _ in range(self.n)]
        self.train_primary = [None for _ in range(self.n)]
        self.test_primary = [None for _ in range(self.n)]
        self.primary_predictions = [None for _ in range(self.n)]
        self.results = []

        # For each dataset, fit and predict the data
        logging.info("Training and predicting for primary classifiers")
        for i, d in enumerate(self.all_data):
            # Create a random forest classifier
            self.primary_classifier[i] = RandomForestClassifier(
                n_estimators=CLASSIFIER_PARAMS["n_estimators"],
                criterion=CLASSIFIER_PARAMS["criterion"],
                min_samples_leaf=CLASSIFIER_PARAMS["min_samples_leaf"],
                max_depth=CLASSIFIER_PARAMS["max_depth"],
                max_features=CLASSIFIER_PARAMS["max_features"],
                oob_score=CLASSIFIER_PARAMS["oob_score"],
                random_state=CLASSIFIER_PARAMS["random_state"]
            )
            # Configuration data for output results
            self.metadata[i] = d["config"]
            # Data for primary test case (remains unchanged for all MRs)
            self.train_primary[i] = d["data"]["train"]
            self.test_primary[i] = d["data"]["test"]
            # Fit the primary classifier
            self.primary_classifier[i].fit(X=self.train_primary[i][0],
                                           y=self.train_primary[i][1])
            # Predict the results for primary classifier
            self.primary_predictions[i] = self.primary_classifier[i].predict(
                self.test_primary[i][0]
            )
        logging.info("Finished training and predicting for primary "
                     "classifiers")

    def tearDown(self):
        df = pd.DataFrame(self.results, columns=DF_COLS, dtype=np.int64)
        df.to_csv(OUT, index=False)
        logging.info("Wrote data to {}".format(OUT))


    def test_mr1(self):
        """
        MR1 -- Affine transformation
        """
        for i in range(self.n):
            results = copy.copy(self.metadata[i])
            transformation_columns = mr.get_transformation_cols(
                self.test_primary[i][0],
                n_transform=7
            )
            # Arbitrary transformation
            m = uniform(-1, 1)
            b = uniform(-1, 1)
            # Generate the follow-up data by applying a linear transformation
            train_followup = mr.mr_linear_transform(self.train_primary[i],
                                                    transformation_columns,
                                                    m=m, b=b)
            test_followup = mr.mr_linear_transform(self.test_primary[i],
                                                   transformation_columns,
                                                   m=m, b=b)
            # Create a follow-up classifier, fit it, and make predictions
            followup_classifier = copy.deepcopy(self.primary_classifier[i])
            followup_classifier.fit(X=train_followup[0], y=train_followup[1])
            follow_up_predictions = followup_classifier.predict(
                test_followup[0]
            )
            # Get the results
            n_diff = 0
            for p, f in zip(self.primary_predictions[i], follow_up_predictions):
                if p != f:
                    n_diff += 1
            logging.info("Number different for run {} of MR 1 was {}"
                         .format(i, n_diff))
            results["n_diff"] = n_diff
            results["mr"] = 1
            with self.subTest(i=i):
                self.assertEqual(n_diff, 0)
            # Add results to self results
            self.results.append(copy.copy(results))
