# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 18:33:18 2023

@author: kaike
"""

# Import libraries
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import statistics as st
import numpy as np

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Import the Grid Search
from sklearn.model_selection import GridSearchCV
# RandomizedSearch for tuning (possibly faster than GridSearch)
from sklearn.model_selection import RandomizedSearchCV

# Import models
import pmdarima as pm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR

# Including to the path another fold
import sys
sys.path.append(r'Models')
from model_lssvr import LSSVR

# Including to the path another fold
sys.path.append(r'Functions')
# Import the serie generator
from MackeyGlassGenerator import MackeyGlass


#-----------------------------------------------------------------------------
# Generate the time series
#-----------------------------------------------------------------------------

Serie = "MackeyGlass"

# The theory
# Mackey-Glass time series refers to the following, delayed differential equation:
    
# dx(t)/dt = ax(t-\tau)/(1 + x(t-\tau)^10) - bx(t)


# Input parameters
a        = 0.2;     # value for a in eq (1)
b        = 0.1;     # value for b in eq (1)
tau      = 17;		# delay constant in eq (1)
x0       = 1.2;		# initial condition: x(t=0)=x0
sample_n = 6000;	# total no. of samples, excluding the given initial condition

# MG = mackey_glass(N, a = a, b = b, c = c, d = d, e = e, initial = initial)
MG = MackeyGlass(a = a, b = b, tau = tau, x0 = x0, sample_n = sample_n)

def Create_Leg(data, ncols, leg, leg_output = None):
    X = np.array(data[leg*(ncols-1):].reshape(-1,1))
    for i in range(ncols-2,-1,-1):
        X = np.append(X, data[leg*i:leg*i+X.shape[0]].reshape(-1,1), axis = 1)
    X_new = np.array(X[:,-1].reshape(-1,1))
    for col in range(ncols-2,-1,-1):
        X_new = np.append(X_new, X[:,col].reshape(-1,1), axis=1)
    if leg_output == None:
        return X_new
    else:
        y = np.array(data[leg*(ncols-1)+leg_output:].reshape(-1,1))
        return X_new[:y.shape[0],:], y

# Defining the atributes and the target value
X, y = Create_Leg(MG, ncols = 4, leg = 6, leg_output = 85)

# Spliting the data into train and test
X_train, X_test = X[201:3201,:], X[5001:5501,:]
y_train, y_test = y[201:3201,:], y[5001:5501,:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_test = scaler.transform(X_test)

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.show()


#-----------------------------------------------------------------------------
# ARIMA
#-----------------------------------------------------------------------------

Model = "ARIMA"

# Define Grid Search parameters

# Optimize parameters
ar = pm.auto_arima(y_train, trace=True)

# Make predictions
y_pred = ar.predict(y_test.shape[0])
    
# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)


arima = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f}'


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()


#-----------------------------------------------------------------------------
# ARIMAX
#-----------------------------------------------------------------------------

Model = "ARIMAX"

# Define Grid Search parameters

# Train with exogenous variables
ar = pm.auto_arima(y_train, exogenous = X_train, trace=True)
# Train the model
ar.fit(y_train, exogenous = X_train)

# # Make predictions
y_pred = ar.predict(n_periods = X_test.shape[0], exogenous = X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)


arimax = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f}'


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()

#-----------------------------------------------------------------------------
# KNN
#-----------------------------------------------------------------------------

Model = "KNN"

# Define Grid Search parameters
parameters = {'n_neighbors': [2, 3, 5, 10], 'weights': ('uniform', 'distance'), 'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'), 'leaf_size': [2, 3, 5, 10, 30, 50], 'p': [1,2]}

# Optimize parameters
kNN = KNeighborsRegressor()
reg = GridSearchCV(kNN, parameters)


reg.fit(X_train,y_train)
reg.best_params_
      
# Make predictions
y_pred = reg.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)


knn = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f}'


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()

#-----------------------------------------------------------------------------
# Regression Tree
#-----------------------------------------------------------------------------

Model = "Regression Tree"

# Define Grid Search parameters
parameters = {'max_depth': [2, 3, 5, 10, 20, 50, 100]}

# Optimize parameters
dt = DecisionTreeRegressor()
reg = GridSearchCV(dt, parameters)
reg.fit(X_train,y_train)
reg.best_params_
      
# Make predictions
y_pred = reg.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)


regression_tree = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f}'


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()

#-----------------------------------------------------------------------------
# Random Forest
#-----------------------------------------------------------------------------

Model = "Random Forest"

# Define Grid Search parameters
parameters = {'n_estimators': [50, 100, 200, 500],'max_features': ('sqrt', 'log2'),'max_depth': [4, 5, 6, 7, 8]}

# Optimize parameters
rf = RandomForestRegressor()
reg = RandomizedSearchCV(rf, parameters)
reg.fit(X_train,y_train.flatten())
reg.best_params_
      
# Make predictions
y_pred = reg.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)


random_forest = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f}'


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()

#-----------------------------------------------------------------------------
# SVM
#-----------------------------------------------------------------------------

Model = "SVM"

# Define Grid Search parameters
#parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1, 2, 3, 5, 10], 'gamma': ['scale', 'auto', 1e-7, 1e-4],'epsilon':[0.05,0.1,0.2,0.3,0.5],'shrinking':[False,True]}
parameters = {'kernel': ('linear', 'rbf', 'sigmoid'), 'C':[1, 10], 'epsilon':[0.1,0.5],'shrinking':[True]}

# Optimize parameters for the centre
svr = SVR()
reg = GridSearchCV(svr, parameters)
reg.fit(X_train,y_train.flatten())
reg.best_params_
      
# Make predictions
y_pred = reg.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)


svm = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f}'


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()

#-----------------------------------------------------------------------------
# LS-SVM
#-----------------------------------------------------------------------------

Model = "LS-SVM"

# Define Grid Search parameters

# Optimize parameters for the centre
lssvr = LSSVR(kernel='linear')
lssvr.fit(X_train, y_train)
      
# Make predictions
y_pred = lssvr.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)


lssvm = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f}'


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()


#-----------------------------------------------------------------------------
# Gradient Boosting
#-----------------------------------------------------------------------------

Model = "GB"

# Define Grid Search parameters
parameters = {'loss':('absolute_error', 'huber', 'quantile'), 'learning_rate':[0.1, 0.3, 0.5, 0.7]}

# Optimize parameters for the centre
GB = GradientBoostingRegressor()
reg = RandomizedSearchCV(GB, parameters)
reg.fit(X_train,y_train.flatten())
reg.best_params_
      
# Make predictions
y_pred = reg.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)


GB = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f}'


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()


#-----------------------------------------------------------------------------
# Light GBM Regressor
#-----------------------------------------------------------------------------

Model = "LGBM"

# Define Grid Search parameters
#parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1, 2, 3, 5, 10], 'gamma': ['scale', 'auto', 1e-7, 1e-4],'epsilon':[0.05,0.1,0.2,0.3,0.5],'shrinking':[False,True]}
parameters = {'boosting_type':('gbdt','dart'),'learning_rate':[0.1,0.3,0.5,0.7,0.9]}

# Optimize parameters for the centre
lgbm = LGBMRegressor()
reg = GridSearchCV(lgbm, parameters)
reg.fit(X_train,y_train.flatten())
reg.best_params_
      
# Make predictions
y_pred = reg.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", RMSE)
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)


LGBM = f'{Model} & {RMSE:.5f} & {NDEI:.5f} & {MAE:.5f}'


# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, linewidth = 5, color = 'red', label = 'Actual value')
plt.plot(y_pred, linewidth = 5, color = 'blue', label = 'Predicted value')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper left')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()

#-----------------------------------------------------------------------------
# Print results
#-----------------------------------------------------------------------------

print(f"\n\n{arima}")
print(f"\n{arimax}")
print(f"\n{knn}")
print(f"\n{regression_tree}")
print(f"\n{random_forest}")
print(f"\n{svm}")
print(f"\n{lssvm}")
print(f"\n{GB}")
print(f"\n{LGBM}")
