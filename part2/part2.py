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

# Load Data
train = pd.read_csv("adult.data", header=None)
test = pd.read_csv("adult.test", header=None)

# Pre processing
# 15 features
# 1 (Workclass)
# 3 (Education)
# 5 (Marital-status)
# 6 (Occupation)
# 7 (Relationship)
# 8 (Race)
# 9 (Sex)
# 13 (native-country)
# 14 (salary class)
le = pp.LabelEncoder()

for i in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
    le.fit(train[i])
    train[i] = le.transform(train[i])
    test[i] = le.transform(test[i])


# Create & Train Models
kNN = kn.KNeighborsClassifier()
NaiveBayes = nb.MultinomialNB()
SVM = svm.LinearSVC() # t time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples
DecisionTree = dt.DecisionTreeClassifier()
RandomForest = en.RandomForestClassifier()
AdaBoost = en.AdaBoostClassifier()
GradientBoosting = en.GradientBoostingClassifier()
LinearDA = da.LinearDiscriminantAnalysis()
MLP = nn.MLPClassifier()
LogisticRegression = lm.LogisticRegression()

models = [kNN, NaiveBayes, SVM, DecisionTree, RandomForest, AdaBoost, GradientBoosting, LinearDA, MLP, LogisticRegression]

# train models
for model in models:
    start = time.time()
    model.fit(train.loc[:, :13], train[14])
    print("--- %s training time: %s seconds ---" % (type(model).__name__, time.time() - start))


# Test Models
executionTimes = []
predictions = []
for model in models:
    start = time.time()
    predictions.append(model.predict(test.loc[:, :13]))
    executionTimes.append(round(time.time() - start, 2))
trueValue = test[14]

# Report evaluation metrics
i = 0
metrics = PrettyTable(["Model", "Execution Time", "Accuracy", "Precision", "Recall", "F1-score", "AUC"])
for prediction in predictions:
    accuracy = round(m.accuracy_score(prediction, trueValue), 2)
    precision = round(m.precision_score(prediction, trueValue), 2)
    recall = round(m.recall_score(prediction, trueValue), 2)
    f1Score = round(m.f1_score(prediction, trueValue), 2)

    fpr, tpr, thresholds = m.roc_curve(trueValue, prediction)
    auc = round(m.auc(fpr, tpr), 2)

    metrics.add_row([type(models[i]).__name__, executionTimes[i], accuracy, precision, recall, f1Score, auc])
    i += 1

metrics.sortby = "Accuracy"
print(metrics)

