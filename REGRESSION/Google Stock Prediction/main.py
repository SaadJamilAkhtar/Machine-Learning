# Google Stock Prediction
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import math
import pickle


def get_classifier_prediction(clf, data):
    return clf.predict(data)


def plot_prediction(forecast_set, classifier, accuracy):
    prediction_df = init_df.copy()
    prediction_df.loc[prediction_df.index[-FORECAST_OUT:], LABEL] = forecast_set

    prediction_df.loc[:, 'Date'] = pd.to_datetime(prediction_df['Date'])  # Convert 'Date' column to datetime type
    prediction_df.loc[:, 'prediction_date'] = prediction_df['Date'] + pd.DateOffset(days=FORECAST_OUT)

    plt.figure(figsize=(12, 6))
    plt.plot(prediction_df['prediction_date'], prediction_df[LABEL])
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Graph of Google Close Stock Price over Time BY {classifier}, Accuracy : {accuracy}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.show()


style.use('ggplot')

# READ DATA
init_df = pd.read_csv('./dataset/GOOGL.csv')
# PREPARE DATA
df = init_df[['Open', 'High', 'Low', 'Close', 'Volume']]
df['HL_percent'] = (df['High'] - df['Close']) / df['Close'] * 100
df['Percent_change'] = (df['Close'] - df['Open']) / df['Open'] * 100
df = df[['Close', 'HL_percent', 'Percent_change', 'Volume']]
# ADD LABEL COLUMN

FORECAST_COL = 'Close'
df.fillna(-9999, inplace=True)

# forecast future close based on today features
FORECAST_OUT = int(math.ceil(0.01 * len(df)))

LABEL = 'Future_close'
df[LABEL] = df[FORECAST_COL].shift(-FORECAST_OUT)
init_df[LABEL] = init_df[FORECAST_COL].shift(-FORECAST_OUT)
df.dropna(inplace=True)

# Prepare data
# Define features and labels
X = np.array(df.drop(LABEL, axis=1))
y = np.array(df[LABEL])

# Scaling the features
X = preprocessing.scale(X)

# Getting data without label for prediction testing
X_lately = X[-FORECAST_OUT:]

# create training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Prepare Classifier ( Linear Regression )
clf_LR = LinearRegression()
clf_LR.fit(X_train, y_train)
accuracy_LR = clf_LR.score(X_test, y_test)

print(f"Accuracy Linear Regression : {accuracy_LR}\n\n")

# ### Prepare Classifier ( SVM ) -- Default

# In[198]:


clf_SVM_L = svm.SVR()
clf_SVM_L.fit(X_train, y_train)
accuracy_SVM_L = clf_SVM_L.score(X_test, y_test)

print(f"Accuracy SVM Linear : {accuracy_SVM_L}\n\n")

# ### Prepare Classifier ( SVM ) -- Polynomial

# In[199]:


clf_SVM_P = svm.SVR(kernel='poly')
clf_SVM_P.fit(X_train, y_train)
accuracy_SVM_P = clf_SVM_P.score(X_test, y_test)

print(f"Accuracy SVM Polynomial : {accuracy_SVM_P}\n\n")

plot_prediction(get_classifier_prediction(clf_LR, X_lately), "Linear Regression", accuracy_LR)

plot_prediction(get_classifier_prediction(clf_SVM_L, X_lately), "SVM Linear", accuracy_SVM_L)

plot_prediction(get_classifier_prediction(clf_SVM_P, X_lately), "SVM Polynomial", accuracy_SVM_P)
