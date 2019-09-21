import pandas as pd
import time
import numpy as np
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.neighbors as kn
import sklearn.ensemble as en
import sklearn.tree as dt
import sklearn.svm as svm
import sklearn.neural_network as nn
import sklearn.metrics as m
import warnings
from prettytable import PrettyTable

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load Data
train = pd.read_csv("data/Part 2 - classification/adult.data")
test = pd.read_csv("data/Part 2 - classification/adult.test")
# Pre processing

le = pp.LabelEncoder()



# Create & Train Models
LinearRegression = lm.LinearRegression()
KNeighborsRegression = kn.KNeighborsRegressor()
RidgeRegression = lm.Ridge()
DecisionTree = dt.DecisionTreeRegressor()
RandomForest = en.RandomForestRegressor()
GradientBoosting = en.GradientBoostingRegressor()
SGD = lm.SGDRegressor()
SVR = svm.SVR()
LinearSVR = svm.LinearSVR()
MLP = nn.MLPRegressor()

models = [LinearRegression, KNeighborsRegression, RidgeRegression, DecisionTree, RandomForest, GradientBoosting
          , SGD, SVR, LinearSVR, MLP]

# train models
for model in models:
    start = time.time()
    model.fit(train.loc[:, :'z'], train['price'])
    print("--- %s training time: %s seconds ---" % (type(model).__name__, time.time() - start))

# Test Models
executionTimes = []
predictions = []
for model in models:
    start = time.time()
    predictions.append(model.predict(test.loc[:, :'z']))
    executionTimes.append(round(time.time() - start, 2))
trueValue = test['price']

# Report evaluation metrics
i = 0
metrics = PrettyTable(["Model", "Execution Time", "MSE", "RMSE", "R-Squared", "MAE"])
for prediction in predictions:
    MSE = round(m.mean_squared_error(prediction, trueValue), 2)
    RMSE = round(np.sqrt(MSE), 2)
    RSquared = round(m.r2_score(prediction, trueValue), 2)
    MAE = round(m.mean_absolute_error(prediction, trueValue), 2)

    metrics.add_row([type(models[i]).__name__, executionTimes[i], MSE, RMSE, RSquared, MAE])
    i += 1

print(metrics)

