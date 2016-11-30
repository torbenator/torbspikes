import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.recurrent import GRU, LSTM
from keras.regularizers import l1, activity_l1, l2, activity_l2, l1l2

import xgboost as xgb


low_dim_l1_reg= 0.001
low_dim_l2_reg= 0.001

high_dim_l1_reg= 0.01
high_dim_l2_reg= 0.01


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

    # number of features durastically changes how well regularization works.
    if Xr.shape[0]<200:
        model.add(Dense(1, input_dim=np.shape(Xr)[1], init='uniform', activation='linear',W_regularizer=l1l2(l1=0.001, l2=0.001)))
    else:
        model.add(Dense(1, input_dim=np.shape(Xr)[1], init='uniform', activation='linear',W_regularizer=l1l2(l1=0.001, l2=0.001)))

    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')
    print model.get_weights()[0][0] # prove to me that you're resetting.
    model.fit(Xr, Yr, nb_epoch=10, verbose=verbose)

    if return_model == True:
        return model

    Yr = model.predict(Xr)
    Yt = model.predict(Xt)


    return Yr, Yt, model.get_weights()


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
    if Xr.shape[1]<200:
        model.add(Dense(25, input_dim=np.shape(Xr)[1], init='uniform', activation='tanh',W_regularizer=l1l2(l1=0.001, l2=0.001)))
    else:
        model.add(Dense(50, input_dim=np.shape(Xr)[1], init='uniform', activation='tanh',W_regularizer=l2(0.001)))
        model.add(Dropout(0.4))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    print model.get_weights()[0][0] # prove to me that you're resetting.

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
    if Xr.shape[1]<200:
        model.add(Dense(100, input_dim=np.shape(Xr)[1], init='uniform', activation='tanh',W_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='tanh'))
        model.add(Dropout(0.2))
    else:
        model.add(Dense(200, input_dim=np.shape(Xr)[1], init='uniform', activation='tanh',W_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation='tanh'))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    model.fit(Xr, Yr, nb_epoch=10,verbose=verbose)

    print model.get_weights()[0][0][0] # prove to me that you're resetting.

    if return_model == True:
        return model

    Yr = model.predict(Xr)
    Yt = model.predict(Xt)

    del model # fucking a

    return Yr, Yt


def RNN_poisson(Xr,Yr,Xt,verbose=False, return_model=False,nLSTM=100):
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

    model = Sequential()
    model.add(LSTM(nLSTM, input_dim=Xr.shape[2],input_length=Xr.shape[1],init='uniform',activation='tanh',W_regularizer=l1l2(l1=0.001, l2=0.01)))
    model.add(Dropout(0.4))
    model.add(Dense(1, input_dim=nLSTM, init='uniform', activation='linear'))
    model.add(Lambda(lambda x: np.exp(x)))
    model.compile(loss='poisson', optimizer='adam')

    print model.get_weights()[0][0] # prove to me that you're resetting.

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

    if Xr.shape[1]<200:

        params = {'objective': "count:poisson",
        'eval_metric': "logloss", #optimizing for poisson loss
        'eta': 0.3, #step size shrinkage. larger--> more conservative / less overfitting
        'alpha':low_dim_l1_reg, #l1 regularization
        'lambda':low_dim_l2_reg, #l2 regularizaion
        'gamma':3, # default = 0, minimum loss reduction to further partitian on a leaf node. larger-->more conservative
        'max_depth': max_depth,
        'seed': 16,
        'silent': 1,
        'missing': '-999.0',
        'colsample_bytree':.5 #new
        }

    else:

        params = {'objective': "count:poisson",
        'eval_metric': "logloss", #optimizing for poisson loss
        'eta': 0.3, #step size shrinkage. larger--> more conservative / less overfitting
        'alpha':0.0005, #l1 regularization
        'lambda':0.01, #l2 regularizaion
        'gamma':4, # default = 0, minimum loss reduction to further partitian on a leaf node. larger-->more conservative
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
    cleaned_scores = np.zeros(Xr.shape[1])
    for i in xrange(Xr.shape[1]):
        if 'f'+str(i) in scores.keys():
            cleaned_scores[i] = scores['f'+str(i)]


    return np.expand_dims(Yr,1), np.expand_dims(Yt,1), cleaned_scores


def XGB_ensemble(Xr, Yr, Xt,return_model=False,max_depth=5):
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
    'max_depth': max_depth,
    'sub_sample': 0.9,
    #'colsample_bytree': 0.75,
    'eta': 0.3, #step size shrinkage. larger--> more conservative / less overfitting
    #'alpha':0.5, #l1 regularization
    #'lambda':0.5, #l2 regularizaion
    'gamma': 100, # default = 0, minimum loss reduction to further partitian on a leaf node. larger-->more conservative
    #'min_child_weight': 4,
    'seed': 16,
    'silent': 1,
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


