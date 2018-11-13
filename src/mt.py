"""
AUTHOR: Brad West
CREATED ON: 2018-11-10
---
DESCRIPTION: Implementation of metamorphic relations for random forests.
"""

from random import uniform, sample
import copy
from sklearn.ensemble import RandomForestClassifier

# TODO Add num predictions

class MetamorphicTester:

    data_test_followup = None
    data_train_followup = None
    predict_primary = None
    predict_followup = None

    def __init__(self, data, params):
        """
        Create a Metamorphic Tester instance

        :param clf: A classifier object
        :param params: passed to RandomForestClassifier
        """
        self.clf_primary = RandomForestClassifier(n_estimators = params["n_estimators"],
                                                  criterion = params["criterion"],
                                                  min_samples_leaf = params["min_samples_leaf"],
                                                  max_depth = params["max_depth"],
                                                  max_features = params["max_features"],
                                                  oob_score = params["oob_score"],
                                                  random_state = params["random_state"])
        self.clf_followup = copy.deepcopy(self.clf_primary)

        self.data_test_primary = data["data"]["test"]
        self.data_train_primary = data["data"]["train"]

        self.metadata = data["config"]
        # TODO Why predictions not transfer
        self.results = dict(n_diff=None,
                            predictions=dict(primary=self.predict_primary,
                                             follow_up=self.predict_followup),
                            config=data["config"])


    def construct_followup(self):
        pass

    def fit(self):
        self.clf_primary.fit(self.data_train_primary[0], self.data_train_primary[1])
        self.clf_followup.fit(self.data_train_followup[0], self.data_train_followup[1])

    def predict(self):
        self.predict_primary = self.clf_primary.predict(self.data_test_primary[0])
        self.predict_followup = self.clf_followup.predict(self.data_test_followup[0])

    def compare_predictions(self):
        n_diff = 0
        for p, f in zip(self.predict_primary, self.predict_followup):
            if p != f:
                n_diff += 1
        self.results["n_diff"] = n_diff

    def get_results(self):
        return copy.deepcopy(self.results)


class MTLinearTransform(MetamorphicTester):

    def construct_followup(self, n_transform=2, m=None, b=None):
        transformation_cols = self.get_transformation_cols(n_transform)
        m = uniform(-1, 1)
        b = uniform(-1, 1)
        self.data_train_followup = self.mr_linear_transform("train",
                                                            transformation_cols,
                                                            m, b)
        self.data_test_followup = self.mr_linear_transform("test",
                                                           transformation_cols,
                                                           m, b)

    def mr_linear_transform(self,
                            data,
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

        if data == "train":
            data = self.data_train_primary
        elif data == "test":
            data = self.data_test_primary
        else:
            raise ValueError("data argument must be one of 'train' or 'test'")

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

    def get_transformation_cols(self, n_transform=2):
        """
        Returns a list of columns to transform

        :param n_transform: int, the number of columns to transform
        :return:
        """
        if n_transform > len(self.data_test_primary[0][0]):
            old = n_transform
            transformed = len(self.data_test_primary[0][0])
            print("Number of columns to transform is greater than columns in "
                  "dataset. Changing parameter from {0} to {1}"
                  .format(old, transformed))
        return sample(range(0, n_transform), n_transform)


class MTAddUninformativeVar(MetamorphicTester):

    def construct_followup(self):
        pass


class MTModifyPredictorOrder(MetamorphicTester):

    def construct_followup(self):
        pass


class MTDuplicateDataset(MetamorphicTester):

    def construct_followup(self):
        pass
