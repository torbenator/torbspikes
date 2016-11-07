import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.recurrent import GRU, LSTM
from keras.regularizers import l1, activity_l1, l2, activity_l2

import xgboost as xgb

#l1_reg = 0.3
l2_reg = 0.001

u_reg=l2_reg

def GLM_poisson(Xr, Yr, Xt,verbose=False,return_model=False):

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

    try:
        model
        del model
        print 'weird model present'
    except NameError:
        None


    model = Sequential()
    model.add(Dense(1, input_dim=np.shape(Xr)[1], init='uniform', activation='linear',W_regularizer=l2(l2_reg)))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    model.fit(Xr, Yr, nb_epoch=3, verbose=verbose)

    if return_model == True:
        return model

    Yr = model.predict(Xr)
    Yt = model.predict(Xt)


    return Yr, Yt, model.get_weights()

def GLM_poisson2(Xr, Yr, Xt,verbose=False,return_model=False):

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

    try:
        model
        del model
        print 'weird model present'
    except NameError:
        None


    model = Sequential()
    model.add(Dense(1, input_dim=np.shape(Xr)[1], init='uniform', activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    model.fit(Xr, Yr, nb_epoch=3, verbose=verbose)

    if return_model == True:
        return model

    Yr = model.predict(Xr)
    Yt = model.predict(Xt)

    del model # fucking a

    return Yr, Yt

def GLM_poisson3(Xr, Yr, Xt,verbose=False,return_model=False):

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

    try:
        model
        del model
        print 'weird model present'
    except NameError:
        None


    model = Sequential()
    model.add(Dense(1, input_dim=np.shape(Xr)[1], init='uniform', activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    model.fit(Xr, Yr, nb_epoch=3, verbose=verbose)

    if return_model == True:
        return model

    Yr = model.predict(Xr)
    Yt = model.predict(Xt)

    del model # fucking a

    return Yr, Yt


def NN_poisson(Xr, Yr, Xt,verbose=False, return_model=False):
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

    try:
        model
        del model
        print 'weird model present'
    except NameError:
        None

    if np.ndim(Xr)==1:
        Xr = np.transpose(np.atleast_2d(Xr))

    model = Sequential()
    model.add(Dense(100, input_dim=np.shape(Xr)[1], init='uniform', activation='tanh',W_regularizer=l2(l2_reg)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    model.fit(Xr, Yr, nb_epoch=10,verbose=verbose)

    if return_model == True:
        return model

    Yr = model.predict(Xr)
    Yt = model.predict(Xt)

    del model # fucking a

    return Yr, Yt


def NN_poisson_2l(Xr, Yr, Xt, verbose=False, return_model=False):
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

    try:
        model
        del model
        print 'weird model present'
    except NameError:
        None

    model = Sequential()
    model.add(Dense(100, input_dim=np.shape(Xr)[1], init='uniform', activation='tanh',W_regularizer=l2(l2_reg)))
    model.add(Dropout(0.4))
    model.add(Dense(50, activation='tanh',W_regularizer=l2(l2_reg)))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    model.fit(Xr, Yr, nb_epoch=10,verbose=verbose)

    if return_model == True:
        return model

    Yr = model.predict(Xr)
    Yt = model.predict(Xt)

    del model # fucking a

    return Yr, Yt


def RNN_poisson(Xr,Yr,Xt,verbose=False, return_model=False):
    """
    Recurrent neural net that feeds into a GLM and returns predictions for a test set

    Parameters:
    ==========
    Xr : features of train set
    Yr : prediction variable of train set
    Xt : features of test set

    Returns:
    ==========
    Yt : predictions of test set

    """

    assert np.ndim(Xr) >2, "Train set  is too small (%r) for this RNN. Make sure you set RNN_out to True in organize_data" % np.ndim(Xr)

    try:
        model
        del model
        print 'weird model present'
    except NameError:
        None

    nLSTM = 20
    model = Sequential()
    model.add(LSTM(nLSTM, input_dim=Xr.shape[2],input_length=Xr.shape[1],init='uniform',activation='tanh',W_regularizer=l2(l2_reg),U_regularizer=l2(u_reg)))
    model.add(Dense(1, input_dim=nLSTM, init='uniform', activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')
    model.fit(Xr, Yr, nb_epoch=10, batch_size=16,verbose=verbose)

    if return_model == True:
        return model

    Yr = model.predict(Xr)
    Yt = model.predict(Xt)

    del model # fucking a

    return Yr, Yt


def XGB_poisson(Xr, Yr, Xt,return_model=False,max_depth=2):
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

    try:
        model
        del model
        print 'weird model present'
    except NameError:
        None

    if np.ndim(Yr)<2:
        Yr = np.expand_dims(Yr,1)

    params = {'objective': "count:poisson",
    'eval_metric': "logloss", #optimizing for poisson loss
    'eta': 0.3, #step size shrinkage. larger--> more conservative / less overfitting
    #'alpha':l1_reg, #l1 regularization
    'lambda':l2_reg, #l2 regularizaion
    'gamma': 1, # default = 0, minimum loss reduction to further partitian on a leaf node. larger-->more conservative
    'max_depth': max_depth,
    'seed': 16,
    'silent': 1,
    'missing': '-999.0',
    'colsample_bytree':.5 #new
    }

    dtrain = xgb.DMatrix(Xr, label=Yr)

    dtest = xgb.DMatrix(Xt)
    dtrain_y = xgb.DMatrix(Xr)

    num_round = 200
    model = xgb.train(params, dtrain, num_round)

    if return_model == True:
        return model

    Yr = model.predict(dtrain_y)
    Yt = model.predict(dtest)

    scores = model.get_score(importance_type='gain')
    cleaned_scores = np.zeros(Xr.shape[0])
    for i in xrange(Xr.shape[0]):
        if 'f'+str(i) in scores.keys():
            cleaned_scores[i] = scores['f'+str(i)]


    return np.expand_dims(Yr,1), np.expand_dims(Yt,1), cleaned_scores


def XGB_ensemble(Xr, Yr, Xt,return_model=False,max_depth=1):
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

    try:
        model
        del model
        print 'weird model present'
    except NameError:
        None

    if np.ndim(Yr)<2:
        Yr = np.expand_dims(Yr,1)

    params = {'objective': "count:poisson",
    'eval_metric': "logloss", #optimizing for poisson loss
    'eta': 0.3, #step size shrinkage. larger--> more conservative / less overfitting
    #'alpha':l1_reg, #l1 regularization
    'lambda':l2_reg, #l2 regularizaion
    'gamma': 1, # default = 0, minimum loss reduction to further partitian on a leaf node. larger-->more conservative
    'max_depth': max_depth,
    'seed': 16,
    'silent': 1,
    'missing': '-999.0'
    }

    dtrain = xgb.DMatrix(Xr, label=Yr)

    dtest = xgb.DMatrix(Xt)
    dtrain_y = xgb.DMatrix(Xr)

    num_round = 200
    model = xgb.train(params, dtrain, num_round)

    if return_model == True:
        return model

    Yr = model.predict(dtrain_y)
    Yt = model.predict(dtest)

    return np.expand_dims(Yr,1), np.expand_dims(Yt,1)


