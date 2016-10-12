import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import Lambda
from keras.layers.recurrent import LSTM
from keras.regularizers import l1, activity_l1

import xgboost as xgb

from sklearn.metrics import accuracy_score



def GLM_poisson(Xr, Yr, Xt,return_model = False):

    """
    Builds a generalized linear model and returns predictions for a test set

    Parameters:
    ==========
    Xr : features of train set
    Yr : prediction variable of train set
    Xt : features of test set

    Returns:
    ==========
    Yt : predictions of test set

    """

    model = Sequential()
    model.add(Dense(1, input_dim=np.shape(Xr)[1], init='uniform', activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='rmsprop')

    model.fit(Xr, Yr, nb_epoch=3, batch_size=16, verbose=False)

    Yt = model.predict(Xt, verbose=False)

    if return_model:
        return Yt, model

    return Yt


def NN_poisson(Xr, Yr, Xt, layers=1, return_model=False):
    """
    Builds a neural net and returns predictions for a test set

    Parameters:
    ==========
    Xr : features of train set
    Yr : prediction variable of train set
    Xt : features of test set

    Returns:
    ==========
    Yt : predictions of test set

    """
    if np.ndim(Xr)==1:
        Xr = np.transpose(np.atleast_2d(Xr))

    model = Sequential()
    model.add(Dense(10, input_dim=np.shape(Xr)[1], W_regularizer=l1(0.01), init='uniform', activation='tanh'))
    model.add(Dropout(0.6))
    if layers == 2:
        model.add(Dense(10, init='uniform', activation='tanh'))
        model.add(Dropout(0.6))

    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    model.fit(Xr, Yr, nb_epoch=3, batch_size=32)

    result = model.predict(Xt)
    if return_model:
        return result, model
    return result

def RNN_poisson(Xr,Yr,Xt, return_model=False):
    """
    Builds a recurrent neural net and returns predictions for a test set

    Parameters:
    ==========
    Xr : features of train set
    Yr : prediction variable of train set
    Xt : features of test set

    Returns:
    ==========
    Yt : predictions of test set

    """

    assert np.ndim(Xr) >2, "Too few dimentions of train set (%r) is too small for this RNN. Make sure you set RNN_out to True in organize_data" % np.ndim(Xr)

    model = Sequential()
    model.add(LSTM(10, input_dim=196))
    model.add(Dense(1, input_dim=10, init='uniform', activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(Xr, Yr, nb_epoch=10, batch_size=16, verbose=2)
    result = model.predict(Xt)

    if return_model:
        return result, model

    return result

def XGB_poisson(Xr, Yr, Xt, params=None):
    """
    Trains a gradient boosted random forest and returns predictions for a test set

    Parameters:
    ==========
    Xr : features of train set
    Yr : prediction variable of train set
    Xt : features of test set

    Returns:
    ==========
    Yt : predictions of test set

    """

    if params == None:

        # pavanparams
        # param = {'objective': "count:poisson",
        # 'eval_metric': "logloss",
        # 'num_parallel_tree': 2,
        # 'eta': 0.07,
        # 'gamma': 1, # default = 0
        # 'max_depth': 1,
        # 'subsample': 0.5,
        # 'seed': 2925,
        # 'silent': 1,
        # 'missing': '-999.0'}

        params = {'objective': "count:poisson",
        'eval_metric': "logloss"}

    dtrain = xgb.DMatrix(Xr, label=Yr)
    dtest = xgb.DMatrix(Xt)

    num_round = 200
    bst = xgb.train(params, dtrain, num_round)

    Yt = bst.predict(dtest)
    return np.expand_dims(Yt,1)


def XGB_ensemble(ensemble_preds_full,ensemble_test_full):
    """
    Uses gradient boosting ensemble method to predict spiking from multiple models
    """

    # split predictions and test into cross validation sets
    cross_validation_ind = ensemble_test_full.shape[0]/2

    ensemble_train = ensemble_preds_full[:cross_validation_ind,:]
    ensemble_test = ensemble_test_full[:cross_validation_ind]
    ensemble_validate_train = ensemble_preds_full[cross_validation_ind:,:]
    ensemble_validate_test = ensemble_test_full[cross_validation_ind:]

    model = xgb.XGBClassifier()
    model.fit(ensemble_train, ensemble_test)
    ensemble_pred = model.predict(ensemble_validate_train)

    accuracy = accuracy_score(ensemble_validate_test,ensemble_pred)

    return ensemble_validate_test,ensemble_pred, accuracy








