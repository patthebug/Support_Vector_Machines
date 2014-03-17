import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import random
import csv
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

train = {}
train['data'] = [[]]
train['target'] = []

with open('letters_training.csv', 'rU') as csvfile:
    training = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in training:
        # row.remove(1025)
        try:
            tempRow = []
            tempRow.append([float(i) for i in row])
            train['data'].append(tempRow[0][0:1023])
            train['target'].append(tempRow[0][1024])
        except:
            continue
train['data'].pop(0)


def resample(data):
    train_data = []
    test_data = []
    train_target = []
    test_target = []
    indexes = range(len(data['data']))
    random.shuffle(indexes)
    for i in range(int(len(indexes)*0.85)):
        train_data.append(data['data'][indexes[i]])
        train_target.append(data['target'][indexes[i]])
    # indexes.reverse()
    for j in range(int(len(indexes)*0.85), len(indexes)):
        test_data.append(data['data'][indexes[j]])
        test_target.append(data['target'][indexes[j]])
    return train_data, train_target, test_data, test_target

train_X, train_Y, test_X, test_Y = resample(train)

print '\nFor kernel=rbf'
C_range = 10.0 ** np.arange(-2, 4)
gamma_range = 10.0 ** np.arange(-5, 3)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(y=train_Y, n_folds=3)
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
grid.fit(train_X, train_Y)
pred = grid.best_estimator_.predict(test_X)
print accuracy_score(test_Y, pred)


print '\nFor kernel=linear'
C_range = 10.0 ** np.arange(-2, 4)
gamma_range = 10.0 ** np.arange(-5, 3)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(y=train_Y, n_folds=3)
grid = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=cv)
grid.fit(train_X, train_Y)
pred = grid.best_estimator_.predict(test_X)
print accuracy_score(test_Y, pred)


f = open('9_54.txt', 'r')
img = []
for line in f:
    for l in range(len(line)):
        if line[l] != '\r' and line[l] != '\n':
            img.append(float(line[l]))
img.pop(0)
clf = svm.SVC(kernel='rbf')
clf.fit(train_X, train_Y)
print '\nPrediction for the new handwritten image: ' + str(int(clf.predict(img)[0]))

train = {}
train['data'] = [[]]
train['target'] = []

with open('letters_training.csv', 'rU') as csvfileReducedDimension:
    trainingReducedDimension = csv.reader(csvfileReducedDimension, delimiter=',', quotechar='|')
    count = 0
    for row in trainingReducedDimension:
        try:
            count += 1
            tempRow = []
            tempRow.append([float(i) for i in row])
            tempRow = np.asarray(tempRow[0][0:1024])
            tempRow = np.reshape(tempRow, (32, 32))
            tempReducedRow = []
            for m in range(0, 32):
                for n in range(7, 25):
                    tempReducedRow.append(tempRow[m][n])
            train['data'].append(tempReducedRow)
            train['target'].append(row[1024])
        except:
            continue

train['data'].pop(0)

train_X, train_Y, test_X, test_Y = resample(train)

print '\nPrediction for images with reduced dimensions'
C_range = 10.0 ** np.arange(-2, 4)
gamma_range = 10.0 ** np.arange(-5, 3)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(y=train_Y, n_folds=3)
grid = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=cv)
grid.fit(train_X, train_Y)
pred = grid.best_estimator_.predict(test_X)
print accuracy_score(test_Y, pred)