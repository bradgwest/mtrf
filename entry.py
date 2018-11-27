"""
AUTHOR: Brad West
CREATED ON: 2018-11-10
---
DESCRIPTION: Entry point for running metamorphic and mutation testing on Random
  Forest implementations.
"""

import pandas as pd
import numpy as np
from make_data import make_data
from mt import *


# Results
OUT = "./data/out.csv"


# Dataset Parameters
N_SAMPLES = [100, 1000, 10000]
N_CLASSES = [(3, 1), (5, 1), (7, 1)]
N_FEATURES = [12]
N_INFO = [(0, 0, 0), (3, 1, 0), (4, 3, 1), (6, 4, 2)]
N_PER = 2


# Classifier Parameters
CLASSIFIER_PARAMS = dict(
    n_estimators = 100,
    criterion = "gini",
    min_samples_leaf = 3,
    max_depth = 15,
    max_features = "auto",
    oob_score = True,
    random_state = 23
)


def main():
    data = make_data(N_SAMPLES, N_CLASSES, N_FEATURES, N_INFO, N_PER, use_seed=True)
    results = run_mt(data)
    df = results_to_df(results)
    df.to_csv(OUT, index=False)
    print("Wrote results dataframe to {}".format(OUT))


def run_mt(data):

    i = 0
    results = [[None for _ in range(0, 4)] for _ in range(0, len(data))]

    for d in data:
        # MR 0
        mr0 = MTLinearTransform(data=d, params = CLASSIFIER_PARAMS)
        mr0.construct_followup(n_transform=7)
        mr0.fit()
        mr0.predict()
        mr0.compare_predictions()
        results[i][0] = mr0.get_results() # TODO Why not subscripted properly?

        # MR 1

        i += 1
        # print("number different: {}".format(mr0.get_results()["n_diff"]))

    return results


def results_to_df(results):
    """
    Results list to data frame for processing

    :param results: list, each element is a given dataset configuration. The
      elements are themselves lists where each element is a metamorphic relation
      result set.
    :return: pandas dataframe
    """
    cols = ["n_samples", "n_features", "n_informative", "n_redundant",
            "n_repeated", "n_classes", "n_clusters_per_class", "seed",
            "mr", "n_diff"]
    df = pd.DataFrame([[None for _ in cols] for _ in range(0, len(results))],
                      columns = cols,
                      dtype=np.int64)

    for i, r in enumerate(results):
        for j, mr in enumerate(r):
            if not mr:
                continue
            c = mr["config"]
            c["n_diff"], c["mr"] = mr["n_diff"], j
            df.iloc[i] = pd.Series(c)

    return df


if __name__ == "__main__":
    main()
