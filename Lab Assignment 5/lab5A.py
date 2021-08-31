"""
    Lab Assignment 5
    Name - Gajraj Singh Chouhan
    Roll No - B19130
    Mobile No - +91-9351159849
"""

import scipy
import pandas as pd
import numpy as np
from math import log, e
from sklearn import mixture, metrics
# imports

data_train = pd.read_csv('seismic-bumps-train.csv', index_col=0) # training data
label_train_data = data_train['class'] # label of training data
data_train.drop(columns=['class'], inplace=True)

class0_train = data_train[label_train_data == 0] # class0
class1_train = data_train[label_train_data == 1] # class1

data_test = pd.read_csv('seismic-bumps-test.csv', index_col=0)
label_test_data = data_test['class'] # label of testing data
data_test.drop(columns=['class'], inplace=True)

# Question 1

print('Question 1\n')

for Q in (2, 4, 8, 16):
    print(f'Q = {Q}')

    gmm_class0 = mixture.GaussianMixture(n_components=Q, random_state=42)
    gmm_class0.fit(class0_train)

    gmm_class1 = mixture.GaussianMixture(n_components=Q, random_state=42)
    gmm_class1.fit(class1_train)

    Pred = [] # our predictions

    for _, row in data_test.iterrows(): # iterating rows
        row = row.values
        res0 = gmm_class0.score_samples(row.reshape(1, 10))

        res1 = gmm_class1.score_samples(row.reshape(1, 10))

        if res0 > res1:
            Pred.append(0)
        else:
            Pred.append(1)


    print(f"Accuracy for GMM Bayes Classifier : {metrics.accuracy_score(Pred, label_test_data)}")
    print("Confusion Matrix : ")
    print(metrics.confusion_matrix(Pred, label_test_data))
    print()

    # break