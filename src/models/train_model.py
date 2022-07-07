import numpy as np
import pandas as pd
import os, sys
# import plotly.express as px

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# RandomOverSampler to handle imbalanced data
from imblearn.over_sampling import RandomOverSampler

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cross Validation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import zscore

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Load data
df = pd.read_csv('src//data//parkinsons.data')
def label_name (row):
       if row['status'] == 1 :
        return 'normal'
       else: return 'patient'

df['label'] = df.apply (lambda row: label_name(row), axis=1)

# print(df.sample(5))
#X = df.drop(columns=['status', 'name'], axis=1)  # Note : dropping column axis = 1; dropping row then axis = 
X = df.iloc[:,1:5]
y = df['label']



# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# random oversampling to handle imbalanced data
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train_org, y_train_org)

# print(len(X_resampled))

def model_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # f1_score = metrics.f1_score(y_test, y_pred)
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    # print('f1 score = {:0.2f}%.'.format(f1_score))

    print('\nClassification Report:\n')
    print(classification_report(y_test, y_pred))
    print("---------------------------------------------\n")

    fig, ax = plt.subplots(figsize=(7, 7))
    print('\nConfusion Matrix:')
    plot_confusion_matrix(model, X_test, y_test,
                          xticks_rotation='horizontal',
                          ax=ax, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.show()

    return

# svm
svm = SVC(kernel='rbf', gamma=0.4, C=1, random_state=0, probability=True)
svm.fit(X_resampled, y_resampled)
model_evaluation(svm, X_test_org, y_test_org)


#Save the model

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)