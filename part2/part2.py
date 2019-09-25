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
from prettytable import PrettyTable

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

# Standardize the inputs
train_mean = train_x.mean()
train_std = train_x.std()
train_x = (train_x - train_mean) / train_std
test_x = (test_x - train_mean) / train_std

# Tricks: add dummy intercept to both train and test ???
train_x['intercept_dummy'] = pd.Series(1.0, index=train_x.index)
test_x['intercept_dummy'] = pd.Series(1.0, index=test_x.index)

# Create & Train Models
kNN = kn.KNeighborsClassifier()
NaiveBayes = nb.MultinomialNB()
SVM = svm.SVC(max_iter=1)
DecisionTree = dt.DecisionTreeClassifier()
RandomForest = en.RandomForestClassifier()
AdaBoost = en.AdaBoostClassifier()
GradientBoosting = en.GradientBoostingClassifier()
LinearDA = da.LinearDiscriminantAnalysis()
MLP = nn.MLPClassifier(max_iter=1)
LogisticRegression = lm.LogisticRegression()

models = [kNN, NaiveBayes, SVM, DecisionTree, RandomForest, AdaBoost, GradientBoosting, LinearDA, MLP, LogisticRegression]

executionTimes = []
predictions = []

for model in models:
    # train model
    start = time.time()
    model.fit(train_x, train_y)

    # Test Model
    predictions.append(model.predict(test_x))
    executionTimes.append(round(time.time() - start, 2))

# Report evaluation metrics
i = 0
metrics = PrettyTable(["Model", "Execution Time", "Accuracy", "Precision", "Recall", "F1-score", "AUC"])
for prediction in predictions:
    accuracy = round(m.accuracy_score(prediction, test_y), 2)
    precision = round(m.precision_score(prediction, test_y), 2)
    recall = round(m.recall_score(prediction, test_y), 2)
    f1Score = round(m.f1_score(prediction, test_y), 2)

    fpr, tpr, thresholds = m.roc_curve(test_y, prediction)
    auc = round(m.auc(fpr, tpr), 2)

    metrics.add_row([type(models[i]).__name__, executionTimes[i], accuracy, precision, recall, f1Score, auc])
    i += 1

print(metrics)

