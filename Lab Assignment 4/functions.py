import numpy as np 
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn import model_selection, metrics, neighbors, preprocessing

def knn(train_data, label_train_data, test_data, label_test_data):

    max_k = 0
    max_accuracy = 0

    for k in (1, 3, 5):
        print(f'For k = {k}\n')
        neigh = neighbors.KNeighborsClassifier(k) # knn classifier
        neigh.fit(train_data, label_train_data) # fitting training data and it's output
        prediction = neigh.predict(test_data) # now predict the test data
        conf_matrix = metrics.confusion_matrix(label_test_data, prediction) # make confusion matrix from prediction and correct
        print(f'Confusion_matrix =\n{conf_matrix}')
        accuracy = metrics.accuracy_score(label_test_data, prediction)
        print(f'Accuracy = {accuracy}\n')

        if accuracy > max_accuracy:
            max_k = k
            max_accuracy = accuracy


    return (max_k, max_accuracy)


def min_max_func(data, min_max_df):
    new = data.copy()
    new_min, new_max = 0, 1
    
    for column in new.columns:
        old_min = min_max_df.loc[column, "min"]
        old_max = min_max_df.loc[column, "max"]
        min_max = lambda val : (((val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min  # min-max normalizing
        new.loc[:, column] = new[column].apply(min_max)

    return new