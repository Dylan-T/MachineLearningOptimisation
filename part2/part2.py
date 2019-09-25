import pandas as pd
import time
import numpy as np
import sklearn.preprocessing as pp
import sklearn.linear_model as lm
import sklearn.neighbors as kn
import sklearn.ensemble as en
import sklearn.tree as dt
import sklearn.svm as svm
import sklearn.neural_network as nn
import sklearn.metrics as m
import sklearn.naive_bayes as nb
import sklearn.discriminant_analysis as da
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# x = [["male", 0], ["female", 1]]
# ohe = pp.OneHotEncoder(sparse=False)
# ohe.fit(x)
# print(ohe.transform(x))

# Load Data
train = pd.read_csv("adult.data", header=None)
test = pd.read_csv("adult.test", header=None)

# Pre processing
# 15 features
# 1 (Workclass) categorical
# 3 (Education) ordinal
# 5 (Marital-status) categorical
# 6 (Occupation) categorical
# 7 (Relationship) categorical
# 8 (Race) categorical
# 9 (Sex) binary
# 13 (native-country) categorical
# 14 (salary class) binary
le = pp.LabelEncoder()

for i in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
    le.fit(train[i])
    train[i] = le.transform(train[i])
    test[i] = le.transform(test[i])


train_x = train.drop([14], axis=1)
train_y = train[14]

test_x = test.drop([14], axis=1)
test_y = test[14]

# Normalize the inputs
x = train_x.values
min_max_scaler = pp.MinMaxScaler()
scaler = min_max_scaler.fit(x)
x_scaled = scaler.transform(x)
train_x = pd.DataFrame(x_scaled)

x = test_x.values
x_scaled = scaler.transform(x)
test_x = pd.DataFrame(x_scaled)

# Tricks: add dummy intercept to both train and test ???
train_x['intercept_dummy'] = pd.Series(1.0, index=train_x.index)
test_x['intercept_dummy'] = pd.Series(1.0, index=test_x.index)


# Create & Train Models
kNN = kn.KNeighborsClassifier()
NaiveBayes = nb.MultinomialNB()
SVM = svm.SVC()
DecisionTree = dt.DecisionTreeClassifier()
RandomForest = en.RandomForestClassifier()
AdaBoost = en.AdaBoostClassifier()
GradientBoosting = en.GradientBoostingClassifier()
LinearDA = da.LinearDiscriminantAnalysis()
MLP = nn.MLPClassifier()
LogisticRegression = lm.LogisticRegression()

models = [kNN, NaiveBayes, SVM, DecisionTree, RandomForest, AdaBoost, GradientBoosting, LinearDA, MLP, LogisticRegression]

executionTimes = []
predictions = []

for model in models:
    # train model
    model.fit(train_x, train_y)

    # Test Model
    predictions.append(model.predict(test_x))

# Report evaluation metrics
i = 0
values = []
for prediction in predictions:
    accuracy = round(m.accuracy_score(prediction, test_y), 2)
    precision = round(m.precision_score(prediction, test_y), 2)
    recall = round(m.recall_score(prediction, test_y), 2)
    f1Score = round(m.f1_score(prediction, test_y), 2)

    fpr, tpr, thresholds = m.roc_curve(test_y, prediction)
    auc = round(m.auc(fpr, tpr), 2)

    values.append((type(models[i]).__name__, accuracy, precision, recall, f1Score, auc))
    i += 1

metrics = pd.DataFrame(values, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "AUC"])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(metrics)
