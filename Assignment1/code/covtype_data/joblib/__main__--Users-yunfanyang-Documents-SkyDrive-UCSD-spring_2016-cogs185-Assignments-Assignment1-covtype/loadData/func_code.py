# first line: 15
@memory.cache
def loadData():
    # import data
    #X = [list(map(int, x.split(',')[:-1])) for x in open('covtype.data').read().splitlines()]
    #_Y = [x.split(',')[-1] for x in open('covtype.data').read().splitlines()]
    #Y = [int(x) - 1 for x in _Y]
    data = np.genfromtxt("covtype.data", delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int32)
    print(X)
    print(Y)
    xTrain = X[:N_TRAIN]
    xTest = X[N_TRAIN:]
    yTrain = Y[:N_TRAIN]
    yTest = Y[N_TRAIN:]
    return xTrain, xTest, yTrain, yTest
