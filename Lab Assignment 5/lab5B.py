"""
    Lab Assignment 5
    Name - Gajraj Singh Chouhan
    Roll No - B19130
    Mobile No - +91-9351159849
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection, linear_model, preprocessing, metrics
#imports

data = pd.read_csv('atmosphere_data.csv')

data_train, data_test = model_selection.train_test_split(data, test_size=0.3, random_state=42, shuffle=True)

data_train.to_csv('atmosphere-train.csv')
data_test.to_csv('atmosphere-test.csv')

# Question 1

print('Question 1\n')

pressure_data_train = data_train['pressure'].values.reshape(-1, 1) # x axis
temp_data_train = data_train['temperature'].values # y axis

pressure_data_test = data_test['pressure'].values.reshape(-1, 1)
temp_data_test = data_test['temperature'].values

regressor = linear_model.LinearRegression() # linear regression
regressor.fit(pressure_data_train, temp_data_train)

temp_train_data_pred = regressor.predict(pressure_data_train) # predicting the train data
RMSE_on_train = metrics.mean_squared_error(temp_data_train, temp_train_data_pred, squared=False) # rmse

print('RMSE Values on Training Data')
print(RMSE_on_train)

temp_test_data_pred = regressor.predict(pressure_data_test) # predicting the test data
RMSE_on_test = metrics.mean_squared_error(temp_data_test, temp_test_data_pred, squared=False)

print('RMSE Values on Test Data')
print(RMSE_on_test)

indices = np.argsort(pressure_data_train, axis=None) # sorting x and y axis before plotting the line graph

plt.scatter(pressure_data_train, temp_data_train, color='dimgrey', marker='x',)
plt.plot(pressure_data_train[indices], temp_train_data_pred[indices], color='red')
plt.title('Linear Reg. | Best Fit Line | Train Data')
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.savefig('PartB-Q1-BestFit.png')
plt.close()


plt.scatter(temp_data_test, temp_test_data_pred,)
plt.title('Linear Reg. | Scatter Plot of Predicted and Original Temperature | Test Data')
plt.xlabel('Original Temperature')
plt.ylabel('Predicted Temperature')
plt.savefig('PartB-Q1-ScatterPlot.png')
plt.close()


# Question 2

print('\nQuestion 2\n')

pressure_data_train = data_train['pressure'].values.reshape(-1, 1) # x axis
temp_data_train = data_train['temperature'].values.reshape(-1, 1) # y axis

pressure_data_test = data_test['pressure'].values.reshape(-1, 1)
temp_data_test = data_test['temperature'].values.reshape(-1, 1)

# storing the training and testing data's rmse values to plot their graph later on
rmse_train = {}
pred_data_train = {}

rmse_test = {}
pred_data_test = {}

for p in (2, 3, 4, 5):

    polynomial_features = preprocessing.PolynomialFeatures(degree=p) # polynomial regression
    x_poly = polynomial_features.fit_transform(pressure_data_train) # getting monomials of polynomials

    regressor = linear_model.LinearRegression() 
    regressor.fit(x_poly, temp_data_train) # using the monomials in the linear regression and fitting

    temp_train_data_pred = regressor.predict(x_poly)
    RMSE_on_train = metrics.mean_squared_error(temp_data_train, temp_train_data_pred, squared=False)

    print(f'degree = {p}\n')
    print('RMSE Values on Train Data')
    print(RMSE_on_train)

    temp_test_data_pred = [regressor.predict(polynomial_features.fit_transform([[i]])).flatten() for i in pressure_data_test.ravel()]
                                # ^^ predicting the data 
    RMSE_on_test = metrics.mean_squared_error(temp_data_test, temp_test_data_pred, squared=False) # rmse for test data

    print('RMSE Values on Test Data')
    print(RMSE_on_test);print()

    rmse_train[p] = RMSE_on_train
    pred_data_train[p] = temp_train_data_pred

    rmse_test[p] = RMSE_on_test
    pred_data_test[p] = temp_test_data_pred

rmse_train = pd.DataFrame.from_dict(rmse_train, orient='index', columns=['RMSE'])
rmse_test = pd.DataFrame.from_dict(rmse_test, orient='index', columns=['RMSE'])

rmse_train.plot.bar(rot=0)
plt.title('Polynomial Reg. | RMSE values of Polynomial Regression | Train Data')
plt.xlabel('Degree of Polynomial')
plt.ylabel('RMSE Values')
plt.legend()
plt.savefig('PartB-Q2-RMSE-Train.png')
plt.close()

rmse_test.plot.bar(rot=0)
plt.title('Polynomial Reg. | RMSE values of Polynomial Regression | Test Data')
plt.xlabel('Degree of Polynomial')
plt.ylabel('RMSE Values')
plt.legend()
plt.savefig('PartB-Q2-RMSE-Test.png')
plt.close()

deg_train = rmse_train.idxmin().to_list()[0] # finding degree of polynomial with minimum rmse 
print(f'Degree of polynomial with minimum rmse for training data = {deg_train}')
pred_data_train = pred_data_train[deg_train] # using the data with minimum rmse
indices = np.argsort(pressure_data_train, axis=None) # sorting x and y axis before plotting the line graph

deg_test = rmse_test.idxmin().to_list()[0] # finding degree of polynomial with minimum rmse 
print(f'Degree of polynomial with minimum rmse for testing data = {deg_test}')
pred_data_test = pred_data_test[deg_test] # using the data with minimum rmse

plt.scatter(pressure_data_train, temp_data_train, label='Original', color='dimgrey', marker='x')
plt.plot(pressure_data_train[indices], pred_data_train[indices], label='Predicted', color='red')
plt.title('Polynomial Reg. | Best Fit Line | Train Data')
plt.xlabel('Pressure')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('PartB-Q2-BestFit-Train.png')
plt.close()

plt.scatter(temp_data_test, pred_data_test)
plt.title('Polynomial Reg. | Original and Predicted Temperature | Test Data')
plt.xlabel('Original Temperature')
plt.ylabel('Predicted Temperature')
plt.savefig('PartB-Q2-ScatterPlot-Test.png')
plt.close()
