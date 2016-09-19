# data preprocessing
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import sklearn.cross_validation as cv
from sklearn.grid_search import GridSearchCV
from collections import Counter
from joblib import Parallel, delayed  
import multiprocessing
num_cores = multiprocessing.cpu_count()

N_TRAIN = 80000
CAP = 100000
def loadData():
    # import data
    X = [list(map(int, x.split(',')[:-1])) for x in open('covtype.data').read().splitlines()[:CAP]]
    _Y = [x.split(',')[-1] for x in open('covtype.data').read().splitlines()[:CAP]]
    Y = [int(x) - 1 for x in _Y]
    
    xTrain, xTest, yTrain, yTest = cv.train_test_split(np.array(X), np.array(Y), train_size = N_TRAIN/len(X), random_state = 13)

    mean = xTrain.mean(axis=0)
    std = xTrain.std(axis=0)
    mean[10:] = 0.0 
    std[10:] = 1.0 
    xTrain = (xTrain - mean) / std 
    xTest = (xTest - mean) / std 

    return xTrain, xTest, yTrain, yTest

def test(estimator, sz):
    print("Trainin with size of " + str(sz))
    estimator.code_size = sz
    estimator.fit(xTrain, yTrain)
    y_pred = estimator.predict(xTest)
    from sklearn.metrics import classification_report
    print ("Code_size: " + str(sz) + "Model accuracy: " + str(accuracy_score(yTest, y_pred)))

#error_output_code for OVAs
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
xTrain, xTest, yTrain, yTest = loadData()

adaboost = AdaBoostClassifier(n_estimators = 128)
randomforest = RandomForestClassifier(n_estimators = 128) 

ESTIMATORS = { 
    'SVM-OVA': OutputCodeClassifier(LinearSVC(loss="squared_hinge", penalty="l2", C=1000, dual=False, tol=1e-3)),
    'BOOSTING-OVA': OutputCodeClassifier(estimator = adaboost),
    'RANDOMFOREST-OVA': OutputCodeClassifier(estimator = randomforest),
}
code_size = {0.2, 0.3, 0.4, 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1, 1.1 , 1.2 , 1.3 , 1.4 , 1.5 , 1.6 , 1.7 , 1.8}

print("Training Classifiers")
print("====================")
for name in ESTIMATORS:
    print("Training %s ... " % name)
    estimator = ESTIMATORS[name]
    Parallel(n_jobs=num_cores, verbose=4)(delayed(test)(estimator, sz) for sz in code_size)

