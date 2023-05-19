import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import math

df = pd.read_csv('./dataset/GOOGL.csv')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df['HL_percent'] = (df['High'] - df['Close']) / df['Close'] * 100
df['Percent_change'] = (df['Close'] - df['Open']) / df['Open'] * 100

df = df[['Close', 'HL_percent', 'Percent_change', 'Volume']]

FORECAST_COL = 'Close'
df.fillna(-9999, inplace=True)

# forecast future close based on today features
FORECAST_OUT = int(math.ceil(0.01 * len(df)))
LABEL = 'Future Volume'
df[LABEL] = df[FORECAST_COL].shift(-FORECAST_OUT)
df.dropna(inplace=True)

# Define our features and labels
X = np.array(df.drop(LABEL, axis=1))
y = np.array(df[LABEL])

# Scaling the features
X = preprocessing.scale(X)

# create training and test sets

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Linear Regression Classifier
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print('Linear regression', accuracy)

# SVM default
clf = svm.SVR()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print('SVM Default', accuracy)

# SVM Polynomial
clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print('SVM Poly', accuracy)
