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

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load Data
data = pd.read_csv("diamonds.csv")

# Pre processing
# remove column 1 (Id's)
data = data.drop([data.columns[0]],  axis='columns')

le = pp.LabelEncoder()

# cut
le.fit(["Fair", "Good", "Very Good", "Premium", "Ideal"])
data['cut'] = le.transform(data['cut'])

# color
le.fit(data['color'])
data['color'] = le.transform(data['color'])

# clarity
le.fit(["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"])
data['clarity'] = le.transform(data['clarity'])

# Split 70:30
train, test = ms.train_test_split(data, test_size=0.3, random_state=309)

train_x = train.drop(['price'], axis=1)
train_y = train['price']
test_x = test.drop(['price'], axis=1)
test_y = test['price']

# Standardize the inputs
train_mean = train_x.mean()
train_std = train_x.std()
train_x = (train_x - train_mean) / train_std
test_x = (test_x - train_mean) / train_std

# Tricks: add dummy intercept to both train and test
train_x['intercept_dummy'] = pd.Series(1.0, index=train_x.index)
test_x['intercept_dummy'] = pd.Series(1.0, index=test_x.index)

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


executionTimes = []
predictions = []
for model in models:
    # train models
    start = time.time()
    model.fit(train_x, train_y)
    # Test Models
    predictions.append(model.predict(test_x))
    executionTimes.append(round(time.time() - start, 2))

# Report evaluation metrics
i = 0
values = []
for prediction in predictions:
    MSE = round(m.mean_squared_error(prediction, test_y), 2)
    RMSE = round(np.sqrt(MSE), 2)
    RSquared = round(m.r2_score(prediction, test_y), 2)
    MAE = round(m.mean_absolute_error(prediction, test_y), 2)

    values.append((type(models[i]).__name__, executionTimes[i], MSE, RMSE, RSquared, MAE))
    i += 1

metrics = pd.DataFrame(values, columns=["Model", "Execution Time", "MSE", "RMSE", "R-Squared", "MAE"])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(metrics)
