"""
        Assignment-2
        Name - Gajraj Singh Chouhan
        Roll No - B19130
        Branch - DSE
        Mobile No - +91-9351159849
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from functions import properties, plot_rmse

# imports

data = pd.read_csv('pima_indians_diabetes_miss.csv')
original_data = pd.read_csv('pima_indians_diabetes_original.csv')

# Question 1

missing_val = data.isna().sum()
# We can count the missing values by using isna(), adding all the cells in a column which would a NaN value (isna() would give each cell 1 or 0 whether it is or is-not a NaN value).
missing_val.plot.bar(rot=0)
# plotting the missing values.
# adding the labels and title.
plt.title("Missing Values in the Attributes")
plt.xlabel("Attributes")
plt.ylabel("Missing Values")
plt.savefig("Q1-Barplot.png")
plt.close()

# Question 2a
print('Question 2a', end='\n\n')

col = data.shape[1] # number of columns
to_drop = 3 # inverse of 1/3rd rows to drop
filtered = data.isna().sum(axis=1)
# finding how many NaN values are in each row by summing the boolean values for each row. (using axis=1 will do sum of each row)
# Now we have a dataframe consisting of number of NaN values in that row 

filtered = filtered[(filtered * to_drop) >= col]
# We can filter the rows to drop, by using the condition that number of NaN values must be greater or equal to 1/3rd of no of columns.
# This will give us True or False Boolean Value if that row is to be dropped or not.

print(f"Total number of tuples deleted = {filtered.size}")

filtered = filtered.index
data = data.drop(filtered, axis=0)
# Now we can drop the rows in the original DataFrame using their indexes we got in the previous step.

print(f"Indices of rows dropped (0-based): ")
print(filtered.to_list());print()

# Question 2b
print('Question 2b', end='\n\n')

filtered = data[data['class'].isna()]
# rows whose 'class' attribute is NaN using conditional filtering
filtered = filtered.index
# getting index of rows which we will delete

print(f"Total number of tuples deleted = {filtered.size}")

data = data.drop(filtered, axis=0) # remove the rows

print(f"Indices of rows dropped (0-based): ")
print(filtered.to_list());print()

# Question 3
print('Question 3', end='\n\n')

count_of_NaN = data.isna().sum()
# again we can use isna() for number of null values.

print("Number of missing values in each attributes: ")
print(count_of_NaN.to_frame().T)
# printing number of missing value per column
# ignore the first 0, it just the first row number as I am displaying the Series in horizontal way.
print(f"Total number of missing values in the dataFrame : {count_of_NaN.sum()}");print()

# Question 4a
print('Question 4a', end='\n\n')
column_mean = data.mean()
data_withmean = data.fillna(column_mean)
# fillna() will fill the NaN values for each attribute by the mean of that column

print('RMSE of Original and Mean Filled data')
plot_rmse(data_withmean, original_data, np.where(pd.isnull(data)), count_of_NaN);print()

# plotting the rmse after replacing the nan values.
plt.title("RMSE of original and mean filled data")
plt.savefig("Q4-MeanFilled.png") 
plt.close()

# Question 4b
print('Question 4b', end='\n\n')

data_with_interpolate = data.interpolate()
# by default each column will be filled with data interpolated with linear method.

print('RMSE of Original and interpolated data')
plot_rmse(data_with_interpolate, original_data, np.where(pd.isnull(data)), count_of_NaN);print()

plt.title("RMSE of original and interpolated data")
plt.savefig("Q4-Interpolated.png")
plt.close()

# Question 4
print('Question 4 (comparing the filled data and original) : ', end='\n\n')

# Comparing the properties (mean, median, mode, standard-dev...) of the 
# original data and after we filled the values in 4a and 4b.
print('Original Data : ')
print(properties(original_data).to_string())
print('\nAfter filling data with mean : ')
print(properties(data_withmean).to_string())
print('\nAfter interpolating the data : ')
print(properties(data_with_interpolate).to_string());print()


# Question 5
print('Question 5 (Outliers)', end='\n\n')
    

data_with_interpolate = data.interpolate()
columns = ["Age", "BMI"]
quartiles = data_with_interpolate.quantile([0.25, 0.5, 0.75]).loc[:, columns]
# quantile() will give the required quartiles (0.25, 0.5, 0.75) for required columns
data.loc[:, columns].plot.box()
plt.title("Box-Plot of data before replacing outliers with median")
plt.savefig("Q5-before-BoxPlot.png")
plt.close()

for col in columns:
    q1, median, q3 = quartiles.loc[:, col].to_list()
    # quartiles for one column
    print(f"Column : {col}")
    check_outlier = lambda num : (q1 - (1.5 * (q3 - q1))) < num < (q3 + (1.5 * (q3 - q1)))
    # this will return boolean value whether a number ("num") would be outlier according to the quartiles.
    for ind, x in enumerate(data_with_interpolate.loc[:, col]):
        # iterating over the column and checking if it's a outlier.
        if not check_outlier(x):
            # if it is a outlier
            data.loc[data_with_interpolate.index[ind], col] = median
            # we can change value of a cell to the median of that column from the column name and row index.
            # data_with_interpolate.index[ind] - row index
            print(x, end=' ')
    print()

data.loc[:, columns].plot.box()
plt.title("Box-Plot of data after replacing outliers with median")
plt.savefig("Q5-after-BoxPlot.png")
plt.close()


print('properties after replacing with median : ')
print(properties(data.loc[:, columns]));print()