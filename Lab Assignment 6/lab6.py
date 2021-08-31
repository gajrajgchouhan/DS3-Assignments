"""
    Name - Gajraj Singh Chouhan
    Roll No - B19130
    Lab Assignment 6
    Mobile No - +91-9351159849
"""

import math
import matplotlib
import numpy as np
import pandas as pd
from sklearn import metrics
import statsmodels.api as sm
import matplotlib.style as style
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

style.use('seaborn')
matplotlib.rcParams['font.sans-serif'] = "DejaVu Sans"
matplotlib.rcParams['font.family'] = "sans-serif"

data = pd.read_csv('datasetA6_HP.csv') # load data

# Question 1
print('Question 1\n')

# Part a

data.plot() # plotting line plot of index and power units
plt.xlabel('Index of the day')
plt.ylabel('Power Units (MU)')
plt.title('Plot of Day and Power Units')
plt.tight_layout()
plt.savefig('Q1-LinePlot.png')
plt.close()

# Part b

def autocorr(x, t=1): # this functions find the correlation original and lagged data
                    # it returns original, lagged data and correlation matrix
                    # t is the time period by which you are lagging the data
                    # e.g. default is `1` which is `1 day` in units of our data
    original = x[t:]
    lag = x[:-t]
    return original, lag, np.corrcoef(np.array([x[:-t], x[t:]]))

power, lag_power, corr = autocorr(data['HP'].values) # autocorrelation of `HP` column
print(f'Pearson correlation (autocorrelation) coefficient between the generated one day lag time sequence and the given time sequence = {corr[0][1]}')

# Part c

plt.scatter(power, lag_power) # plotting the original and lagged value
plt.xlabel('time sequence')
plt.ylabel('one day lagged generated sequence')
plt.title('Scatter Plot of Original and Lag sequence of 1')
plt.tight_layout()
plt.savefig('Q1-ScatterPlot.png')
plt.close()

# Part d

autocor = []
for day in range(1, 7+1): # autocorrelation for lagged day from 1 to 7
    _, _, corr = autocorr(data['HP'].values, t=day)
    autocor.append(corr[0][1])

x = range(1, 8)
plt.plot(x, autocor, marker='o') # plotting the autocorrelation values
plt.xlabel('lagged values')
plt.ylabel('correlation coefficient')
plt.title('Plot of lag value and correlation')
plt.tight_layout()
plt.savefig(f'Q1-AutoCorrPlot.png')
plt.close()

# Part e

sm.graphics.tsa.plot_acf(data['HP'].values, lags=7) # plotting the lag value using plot_acf
plt.tight_layout()
plt.savefig('Q1-plot_acf.png')
plt.title('Plot of lag value and correlation')
plt.close()

# Question 2
print('Question 2\n')

values = pd.DataFrame(data['HP'].values)
dataframe = pd.concat([values.shift(1), values], axis=1) # making a dataframe of original and lagged values by 1

X = dataframe.values

train, test = X[:len(X)-250], X[len(X)-250:] # taking first 250 days as train and rest as test
train_X, train_y = train[:,0], train[:,1] # splitting the train and test as input and expected value
test_X, test_y = test[:,0], test[:,1]

# persistence model
model_persistence = lambda x : x # persistence model will output the input we gave


predictions = []
for x in test_X:
    pred = model_persistence(x)
    predictions.append(pred)

test_score = metrics.mean_squared_error(test_y, predictions, squared=False) # rmse
print(f'Test RMSE: {test_score}')

# Question 3

print('Question 3\n')

X = data['HP'].values
train, test = X[:len(X)-250], X[len(X)-250:]

def autocorr_(day):

    print(f'lag = {day}')
    model = AutoReg(train, lags=day, old_names=False)
    model_fit = model.fit()
    print(f'Coefficients: {model_fit.params}') # coefficient of linear regression

    # make predictions

    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    
    # for i in range(len(predictions)):
        # print(f'predicted={predictions[i]}, expected={test[i]}')
    
    test_score = metrics.mean_squared_error(test, predictions, squared=False)
    print(f'Test RMSE: {test_score}\n')
    
    plt.scatter(test, predictions)
    plt.xlabel('Input Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Input and Predicted Values for day = {day}')
    plt.tight_layout()
    plt.savefig(f'Q3-ScatterPlotday{day}.png')
    plt.close()

for day in (1, 5, 10, 15, 25):
    autocorr_(day)

lag = 1

while True:
    _, _, corr = autocorr(data['HP'], t=lag)

    corr = corr[0][1]

    if not (abs(corr) > 2 / math.sqrt(data['HP'].shape[0])):
        print(f'lag = {lag} is not appropriate')
        lag -= 1
        break

    lag += 1

for day in range(1, lag+1):

    autocorr_(day)