from __future__ import division, print_function

import os
from time import time
import argparse
import numpy as np

from sklearn.datasets import fetch_covtype, get_data_home
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import zero_one_loss
from sklearn.externals.joblib import Memory
from sklearn.utils import check_array
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(os.path.join(get_data_home(), 'covertype_benchmark_data'),
                mmap_mode='r')
@memory.cache
def load_data(dtype=np.float32, order='C', random_state=13):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    ## Load dataset
    print("Loading dataset...")
    data = fetch_covtype(download_if_missing=True, shuffle = True, random_state = random_state)
    X = check_array(data['data'], dtype=dtype, order=order)
    y = (data['target'] - 1).astype(np.int)

    ## Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 522911
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    ## Standardize first 10 features (the numerical ones)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    mean[10:] = 0.0
    std[10:] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test, y_train, y_test

adaboost = AdaBoostClassifier()
randomforest = RandomForestClassifier() 
ESTIMATORS = {
    'SVM-OVA': LinearSVC(loss="squared_hinge", penalty="l2", C=1000, dual=False,
                           tol=1e-3),
#    'SVM-EXPLICIT': LinearSVC(loss="squared_hinge", penalty='l2', C=1000, dual=False, tol = 1e-3, multi_class='crammer_singer'),
#    'BOOSTING-OVA': OneVsRestClassifier(estimator = adaboost),
#    'BOOSTING-EXPLICIT': AdaBoostClassifier(),
#    'RANDOMFOREST-OVA': OneVsRestClassifier(estimator = randomforest),
#    'RANDOMFOREST-EXPLICIT': RandomForestClassifier(),
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifiers', nargs="+",
                        choices=ESTIMATORS, type=str,
                        default=['SVM-EXPLICIT', 'SVM-OVA', 'BOOSTING-OVA','BOOSTING-EXPLICIT', 'RANDOMFOREST-OVA', 'RANDOMFOREST-EXPLICIT'],
                        help="list of classifiers to benchmark.")
    args = vars(parser.parse_args())

    print(__doc__)

    X_train, X_test, y_train, y_test = load_data() 
    GRIDS = {
            'SVM-OVA': {
                            'C': [0.1, 1, 10, 100, 1000, 10000]
                    },
            'SVM-EXPLICIT': {
                            'C': [0.1, 1, 10, 100, 1000, 10000]
                    }, 
            'BOOSTING-OVA': {
                            'estimator__n_estimators': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
                    },
            'BOOSTING-EXPLICIT': {
                            'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
                    },
            'RANDOMFOREST-OVA': {
                            'estimator__n_estimators': [2, 4, 8, 16, 32, 64, 128, 256]
                    },
            'RANDOMFOREST-EXPLICIT': {
                            'n_estimators': [2, 4, 8, 16, 32, 64, 128, 256]
                    },
    }

    print("")
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print("%s %d (pos=%d, neg=%d, size=%dMB)"
          % ("number of train samples:".ljust(25),
             X_train.shape[0], np.sum(y_train == 1),
             np.sum(y_train == 0), int(X_train.nbytes / 1e6)))
    print("%s %d (pos=%d, neg=%d, size=%dMB)"
          % ("number of test samples:".ljust(25),
             X_test.shape[0], np.sum(y_test == 1),
             np.sum(y_test == 0), int(X_test.nbytes / 1e6)))

    print()
    print("Training Classifiers")
    print("====================")
    error, train_time, test_time = {}, {}, {}
    for name in ESTIMATORS: #sorted(args["classifiers"]):
        print("Training %s ... " % name, end="")
        estimator = ESTIMATORS[name]
        estimator_params = estimator.get_params()


        CV = GridSearchCV(estimator, param_grid=GRIDS[name], cv=5, n_jobs=-1 , verbose = 4)

        time_start = time()
        CV.fit(X_train, y_train)
        train_time[name] = time() - time_start

        time_start = time()
        y_pred = CV.predict(X_test)
        test_time[name] = time() - time_start

        error[name] = zero_one_loss(y_test, y_pred)

        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print("%s %s %s %s"
          % ("Classifier  ", "train-time", "test-time", "error-rate"))
    print("-" * 44)
    for name in sorted(args["classifiers"], key=error.get):
        print("%s %s %s %s" % (name.ljust(12),
                               ("%.4fs" % train_time[name]).center(10),
                               ("%.4fs" % test_time[name]).center(10),
                               ("%.4f" % error[name]).center(10)))

    print()
