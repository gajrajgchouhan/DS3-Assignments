import warnings
import numpy as np
import pandas as pd
from collections import Counter

class myKNN:

    """
        My Implementation of KNN classifier.
    """

    def __init__(self, k):

        if (k % 2 == 0):
            w = f"k = {k} is even, it may not work in the KNN Classifier"
            warnings.warn(w)
        
        self.k = k

    def fit(self, train : "DataFrame", label_train : "Series"):

        if train.empty or label_train.empty:

            raise Exception("Cannot use empty dataFrames")

        if train.shape[0] != label_train.shape[0]:

            raise Exception(f"Please input correct dataFrames, rows of DF1 = {train.shape[0]}"
                            f"doesn't match rows of DF2 = {label_train.shape[0]}")

        self.train = train
        self.label_train = label_train

    def _predict(self):
        prediction = []
        for index, row in self.test.iterrows():
            
            subtraction = self.train.sub(row.squeeze()) # subtract the row from every row of train data
                                                        # to then find the euclidean distance
            after_sub = np.sqrt(np.square(subtraction).sum(axis=1)) # finding euclidean distance - sqrt(sum of squares of elements in vector)
            after_sub = pd.concat([after_sub, self.label_train], axis=1) # concat with label data
        
            final = after_sub.sort_values([after_sub.columns[0]]) # sort by distance
            final = final.iloc[:self.k, ] # get the first k neighbors
            final = list(final[self.label_train.name])
            final = Counter(final).most_common(1)[0][0] # most common neighbor
        
            prediction.append(final) # appending our final prediction
        
        return prediction

    def predict(self, test):

        if test.shape[1] != self.train.shape[1]:

            raise Exception("Input same number of columns of training and testing"
                            f"Test Columns = {test.shape[1]}, Training Columns = {self.train.shape[1]}")

        self.test = test
        return self._predict()
