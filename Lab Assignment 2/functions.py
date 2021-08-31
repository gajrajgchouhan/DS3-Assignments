import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt

def properties(data):
    """
    Required for Q4.
    Returns dataFrame of consisting of mean, median, mode & standard-dev of the attributes.
    """
    data_top = data.columns.to_list()
    df = {"mean" : [], "median" : [], "mode" : [], "standard-dev" : []}
    # I'll be using a dataframe to store the properties asked in the question.
    # The properties will be stored as a list for each key in the dictionary
    # We can convert the dictionary to a proper dataFrame later on.
    for column in data_top:
        data_of_attribute = data.loc[:, column]
        # selecting the whole column (:) using the label (column) by loc method
        df["mean"].append(data_of_attribute.mean())
        # mean of column using mean method
        df["median"].append(data_of_attribute.median())
        # median of column using median method
        df["mode"].append(tuple(data_of_attribute.mode().to_list()))
        # mode of column using mode method, using to_list incase there may not be a unique mode
        df["standard-dev"].append(np.std(data_of_attribute))
        # standard deviation of this column using np.std method

    df = pd.DataFrame(df, index=data_top)
    # Now I converted the dictionary to the Data Frame using the pd.DataFrame() function, I will be 
    # using the asked attributes as index of the DataFram
    # print(df.to_string(), end="\n\n")
    # We can print the final dataframe using to_string method to print the full version including  
    # all of columns and rows.

    return df


def plot_rmse(first, original, index_ls, na):
    """
    RMSE = sqrt(sum(diffrence_of_square / number of missing values for that attribute))
    first = for calculating error (filled values)
    original = for calculating error (original data)
    index_ls = index of cells on which we have to calculate error
    na = number of missing values for each attribute
    """
    rmse_of_attri = {col:0 for col in first.columns.to_list()}# Series of column names, as we will plot for each attribute
    cells = zip(*index_ls) # indices
    # number = {}
    for row, col in cells:
        ind = first.columns.to_list()[col] # column name from index
        rmse_of_attri[ind] += ((float(first.loc[first.index[row], ind]) - float(original.loc[first.index[row], ind]))**2) / na[col]
    for key, val in rmse_of_attri.items():
        rmse_of_attri[key] = math.sqrt(rmse_of_attri[key])
    rmse_of_attri = pd.Series(rmse_of_attri)
    # finally taking square root
    print(rmse_of_attri)
    rmse_of_attri.plot.bar(rot=0)
    plt.xlabel("Attributes")
    plt.ylabel("RMSE")
