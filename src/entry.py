"""
AUTHOR: Brad West
CREATED ON: 2018-11-10
---
DESCRIPTION: Entry point for running metamorphic and mutation testing on Random
  Forest implementations.
"""

from generateData import generate_data
from MetamorphicTester import *


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
    data = generate_data(N_SAMPLES, N_CLASSES, N_FEATURES, N_INFO, N_PER, use_seed=True)
    data = [data[0]]  # delete me
    results = run_mt(data)
    print("made here")


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
        results[i][0] = mr0.get_results()

        # MR 1

        i += 1


    return results


def compare_predictions():
    pass


if __name__ == "__main__":
    main()
