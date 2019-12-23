# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 17:17:40 2019

@author: Robert
"""

#!/C:\Users\Robert\Anaconda3\python3

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score

# Load data files
org_returns = pd.read_csv('org_returns.csv', index_col = 0)
org_returns = org_returns.sort_index()

# Split data into training, validation and test sets
org_train, org_validate, org_test = np.split(org_returns,
                                             [int(.6*len(org_returns)), int(.8*len(org_returns))])

# SVR algorithm used for tuning
def model_SVR(data, stocksymbol, window_size, C, hist, epsilon):
    temp = data.copy()
    # Create the label used for prediction
    temp[stocksymbol + '_direction'] = np.where(temp[stocksymbol] >= 0, 1, -1)
    # Get standard deviation of the index
    temp['index_std'] = temp['index'].rolling(hist).std()
    # Get standard deviation of the observed stock
    temp[stocksymbol + '_std'] = temp[stocksymbol].rolling(hist).std()
    
    # The standard deviation calculations create some NaN values so these have to be dropped
    temp = temp[hist - 1:]
    # Initialise lists and numbers
    accuracy = []
    predictions = []
    class_predictions = []
    class_true = []
    auc_score = 0
    
    for i in range(window_size, temp.shape[0]):
        # Select window, including the prediction terms
        total_window = temp[(i - window_size):i + 1]
        # Drop columns where there is not stock data for all days in the window
        total_window = total_window.dropna(axis='columns')
        if stocksymbol in total_window.columns:
            # Select training part
            data_window = total_window[:-1]

            # Split training part into features and labels
            Y = data_window[stocksymbol + '_direction']
            X = data_window.drop(stocksymbol + '_direction', axis=1)
            
            # Get the observation to use for prediction
            to_predict = total_window[-1:]

            # Split prediction part features and labels
            X_to_predict = to_predict.drop(stocksymbol + '_direction', axis=1)
            Y_to_predict = to_predict[stocksymbol + '_direction']

            # Initialise and train SVR model
            model = SVR(gamma='scale', C = C, epsilon = epsilon)
            model.fit(X, Y)
            # Get predictions
            prediction = model.predict(X_to_predict)
            # Classify predictions
            class_prediction = [1] if prediction >= 0 else [-1]
            #  Append to lists
            predictions.append(prediction)
            class_predictions.append(class_prediction)
            class_true.append(Y_to_predict)
            
            # Get accuracy of the predictions
            temp_accuracy = accuracy_score(Y_to_predict, class_prediction)
            accuracy.append(temp_accuracy)
            
    # Get the AUC score but skip it if the stock data did not cover the window size
    if len(temp[stocksymbol].dropna(axis = 0)) > window_size:
        auc_score = roc_auc_score(class_true, class_predictions)
                
    # Return accuracy and auc_score
    return(np.mean(accuracy), auc_score)
    
# Grid search function
def the_gridsearch(data, stocksymbol, window_size, C, hist, epsilon):
    # Create the grid in a dataframe
    params_values = [(x, y, z, t) for x in window_size for y in C 
                     for z in hist for t in epsilon]
    params = pd.DataFrame(data=params_values,# index=params_names, 
                          columns = ['window_size', 'C', 'hist', 'epsilon'])
    # Get the accuracy and auc score from all combination of parameters using the model_SVR function
    for i in params.index:
        #time.sleep(0.25)
        print('Current progress: {}%'.format(round(params.index.get_loc(i)/params.shape[0]*100, 3)))
        params.loc[params.index==i, 'accuracy'], params.loc[params.index==i, 'auc_score'] = model_SVR(
            data, stocksymbol,
            int(params.loc[params.index==i, 'window_size']),
            float(params.loc[params.index==i, 'C']),
            int(params.loc[params.index==i, 'hist']),
            int(params.loc[params.index==i, 'epsilon']))
        
    return params

# Set all parameters to use in gridsearch tuning
window_size = [180, 360]
C = [1.5, 3]
hist = [5, 10, 30]
epsilon = [0.1]

org_train_drop = org_train.dropna(axis = 1, how = 'all')
stocklist = org_train_drop.drop('index', axis = 1).columns

# Load data from previously saved files
best_accuracy = pd.read_csv('best_accuracy.csv', index_col = 0)
best_auc_score = pd.read_csv('best_auc_score.csv', index_col = 0)

# Get list of stocks that have not yet been tuned
unprocessed_stocks = list(set(stocklist) - set(best_auc_score['symbol'].unique()))

counter = 0
for stock in unprocessed_stocks:
    counter += 1
    print('Stockname: ' + stock + ' - Number ' + str(counter) + ' out off ' + str(len(unprocessed_stocks)))
    
    # Run the gridsearch fungion
    results_SVR = the_gridsearch(org_train_drop, stock, window_size, C, hist, epsilon)
    results_SVR['symbol'] = stock
    
    # Take the highest accuracy and AUC combinations
    acctemp = results_SVR.sort_values('auc_score', ascending = False).head(1)
    auc_scoretemp = results_SVR.sort_values('auc_score', ascending = False).head(1)
    
    best_accuracy = pd.concat([best_accuracy, acctemp])
    best_auc_score = pd.concat([best_auc_score, auc_scoretemp])
    
    # Save all data frames to csv files
    results_SVR.to_csv('./data/' + stock + '_results_SVR.csv')
    best_accuracy.to_csv('best_accuracy.csv')
    best_auc_score.to_csv('best_auc_score.csv')

