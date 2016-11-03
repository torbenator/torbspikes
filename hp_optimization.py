import csv
import numpy as np
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.recurrent import GRU, LSTM
from keras.regularizers import l1, activity_l1, l2, activity_l2
from keras.wrappers.scikit_learn import KerasRegressor

import xgboost as xgb

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

import models,utils


nn_param_dict_1l = {
    'dropout' : [0.2,0.4,0.6,0.8],
    'n_neurons' : [10,25,50,75,100],
}

def create_1l_nn_model(n_neurons, dropout,input_dim=196):

    model = Sequential()
    model.add(Dense(n_neurons, input_dim=input_dim, init='uniform', activation='tanh')) #W_regularizer=l2(l2_reg)
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    return model


nn_param_dict_2l = {
    'dropout1' : [0,0.2,0.4,0.6,0.8,0.9],
    'dropout2' : [0,0.2,0.4,0.6,0.8,0.9],
    'n_neurons_1l' : [25,50,75,100],
    'n_neurons_2l' : [3,5,10]
}

def create_2l_nn_model(n_neurons_1l, n_neurons_2l, dropout1,dropout2,input_dim=196):

    model = Sequential()
    model.add(Dense(n_neurons_1l, input_dim=input_dim, init='uniform', activation='tanh'))
    model.add(Dropout(dropout1))
    model.add(Dense(n_neurons_2l, input_dim=input_dim, init='uniform', activation='tanh'))
    model.add(Dropout(dropout2))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    return model


nn_param_dict_2l = {
    'dropout1' : [0,0.2,0.4,0.6,0.8,0.9],
    'dropout2' : [0,0.2,0.4,0.6,0.8,0.9],
    'n_neurons_1l' : [25,50,75,100],
    'n_neurons_2l' : [3,5,10]
}

def create_2l_nn_model(n_neurons_1l, n_neurons_2l, dropout1,dropout2):

    model = Sequential()
    model.add(Dense(n_neurons_1l, input_dim=input_dim, init='uniform', activation='tanh'))
    model.add(Dropout(dropout1))
    model.add(Dense(n_neurons_2l, activation='tanh'))
    model.add(Dropout(dropout2))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    return model

rnn_param_dict = {
    'dropout1' : [0,0.2,0.4,0.6,0.8,0.9],
    'nLSTM' : [5,10,25,50],
}

def create_rnn_model(nLSTM, dropout1,input_dim=196,input_length=5):

    model = Sequential()
    model.add(LSTM(nLSTM, input_dim=input_dim,input_length=input_length,W_regularizer=l1(0.01),activation='tanh'))
    model.add(Dropout(dropout1))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    return model

xgb_param_dict = {
    'dropout1' : [0,0.2,0.4,0.6,0.8,0.9],
    'nLSTM' : [5,10,25,50],
}

def find_hyperparams(build_fn, param_grid, X, y,verbose=False,save_dir=None,fname=None,safe=True):

    model = KerasRegressor(build_fn=build_fn, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X, y)

    if verbose ==  True:
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    if save_dir and fname:
        if os.path.exists(save_dir):
            if safe:
                y = raw_input("Directory already exists. Data may be overwritten. Type y to continue.")
                assert y == 'y'
        else:
            print "Creating " + save_dir
            os.makedirs(save_dir)

        with open(save_dir+fname+'.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile)
            for k,v in grid_result.best_params_.items():
                writer.writerow([k,v])

    return grid_result.best_params_


def cross_validate_params(model, X, y):
    # broken
    seed = 16

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(model, X, y, cv=kfold)

    print("CV score " + str(results.mean()))

    return results
