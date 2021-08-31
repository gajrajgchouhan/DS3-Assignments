"""
    Name - Gajraj Singh Chouhan
    Lab Assignment 4
    Roll No - B19130
    Mobile No - +91-9351159849
"""

import numpy as np
import pandas as pd
from functions import knn, min_max_func
from scipy.stats import multivariate_normal
from sklearn import model_selection, metrics, neighbors, preprocessing
# imports

data = pd.read_csv('seismic_bumps1.csv')
data = data.drop(columns=["nbumps","nbumps2","nbumps3","nbumps4","nbumps5","nbumps6","nbumps7","nbumps89"])

final = {}

# Question 1

print('Question 1\n')

class_0 = data[data['class'] == 0] # separating the classes
class_1 = data[data['class'] == 1]

# splitting each classes into 70% train and 30% test

class0_train, class0_test = model_selection.train_test_split(class_0, test_size=0.3, random_state=42, shuffle=True)
class1_train, class1_test = model_selection.train_test_split(class_1, test_size=0.3, random_state=42, shuffle=True)

train_data = pd.concat([class0_train, class1_train]) # making training data
label_train_data = train_data['class'] # separating label and data

test_data = pd.concat([class0_test, class1_test])
label_test_data = test_data['class']

train_data.to_csv('seismic-bumps-train.csv') # outputing csv
test_data.to_csv('seismic-bumps-test.csv')

for df in (class0_test, class0_train, class1_test, class1_train, train_data, test_data, ):
    df.drop(columns=['class'], inplace=True) # removing label after separation

final["Q1"] = knn(train_data, label_train_data, test_data, label_test_data)

# Question 2 

print('Question 2\n')

# Normalizing train data
normalized_train_data = train_data.copy()
before_normalize = pd.concat([normalized_train_data.min(), normalized_train_data.max(),], axis=1, keys=["min", "max",]) 

normalized_train_data = min_max_func(normalized_train_data, before_normalize)
normalized_train_data.to_csv('seismicbumps-train-Normalised.csv') # output

# Normalizing test data
normalized_test_data = test_data.copy()
normalized_test_label = label_test_data.copy()

# removing rows having element which are not in the range of (min, max) of any column as we would be applying the (min, max) of train data

to_drop = set()
for row in normalized_test_data.itertuples(): # iterating on rows
    ind = 0
    for col in normalized_test_data.columns: # iterating on columns
        if getattr(row, col) < before_normalize.iloc[ind]["min"] or getattr(row, col) > before_normalize.iloc[ind]["max"]:
            to_drop.add(row.Index) # add the index to a set
            break
        ind += 1

normalized_test_data = normalized_test_data.drop(to_drop) # dropping them from test and the label
normalized_test_label = normalized_test_label.drop(to_drop)

normalized_test_data = min_max_func(normalized_test_data, before_normalize)
normalized_test_data.to_csv('seismic-bumps-test-normalised.csv') # output

final["Q2"] = knn(normalized_train_data, label_train_data, normalized_test_data, normalized_test_label)

# Question 3

print('Question 3\n')

mean = [class0_train.mean(), class1_train.mean()] # mean for each class
cov = [class0_train.cov().to_numpy(), class1_train.cov().to_numpy()] # covariance matrix for each class

class0_prior = class0_train.shape[0] / train_data.shape[0] # prior = no of elements of that class / total number of elements
class1_prior = class1_train.shape[0] / train_data.shape[0]
prior = [class0_prior, class1_prior]

likelihood = lambda x, class_ind : multivariate_normal.pdf(x=x, mean=mean[class_ind], cov=cov[class_ind], allow_singular=True ) # likelihood = p(x, mu, sigma)
evidence = lambda x : sum([likelihood(x, class_ind) * prior[class_ind] for class_ind in (0, 1)]) # evidence = sum(likelihood * prior for each class)

prediction = []

for row in test_data.values:
    PostProbs = []
    for class_ind in (0, 1):
        posterior_prob = (likelihood(row, class_ind) * prior[class_ind]) / evidence(row)
        PostProbs.append(posterior_prob)
    if PostProbs[0] > PostProbs[1]:
        prediction.append(0)
    else:
        prediction.append(1)

conf_matrix = metrics.confusion_matrix(label_test_data, prediction) # make confusion matrix from prediction and correct
print(f'Confusion_matrix =\n{conf_matrix}')
accuracy = metrics.accuracy_score(label_test_data, prediction)
print(f'Accuracy = {accuracy}\n')

final["Q3-Bayes"] = (np.nan, accuracy)

print(pd.DataFrame(final, index=['K', 'Accuracy']))