import os
import time
import csv

from collections import OrderedDict
import numpy as np

import bookkeeping, utils
from models import *



def run_funct(funct_string, X_train, X_test, y_train, y_test, include_train=True):
    """
    Calculates the bootstrap statistics for data and a model

    """

    Yr, Yt = globals()[funct_string](X_train,y_train,X_test)
    #Yr, Yt = eval(funct_string)(X_train,y_train,X_test)

    real_r_test, bootstrap_hist_test, pct_score_test = utils.bootstrap_stats(y_test, Yt)

    if include_train:
        real_r_train, bootstrap_hist_train, pct_score_train = utils.bootstrap_stats(y_train, Yr)
        return real_r_train, bootstrap_hist_train, pct_score_train, real_r_test, bootstrap_hist_test, pct_score_test, Yr, Yt

    return real_r_test, bootstrap_hist_test, pct_score_test, Yr, Yt


def run_method(all_spikes, method, save_dir, fname,
    run_subsample=None,
    convolve_params=None, n_wins=10,
    winsize=1, shrink_X=None, flatten_X=False,
    include_my_neuron=False, verbose=False,
    safe=True, note=None):

    """
    Runs 1 method and saves the results in a file
    """

    if os.path.exists(save_dir+fname):
        if safe:
            y = raw_input(save_dir+fname+" already exists. Data may be overwritten. Type y to continue.")
            assert y == 'y'
    else:
        print "Creating " + save_dir+fname
        os.makedirs(save_dir+fname)

    os.chdir(save_dir+fname)

    with open('params.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date and time: ', str(time.strftime("%c"))])
        writer.writerow(['method', str(method)])
        writer.writerow(['neuron subsample: ', str(run_subsample)])
        writer.writerow(['convolve params: ', str(convolve_params)])
        writer.writerow(['n_wins: ', str(n_wins)])
        writer.writerow(['winsize: ', str(winsize)])
        writer.writerow(['shrink_X: ', str(shrink_X)])
        writer.writerow(['note',str(note)])

    train = []
    train_bootstrap = []
    train_pct = []

    test = []
    test_bootstrap = []
    test_pct = []

    train_preds = []
    test_preds = []

    if run_subsample is None:
        runs = xrange(all_spikes.shape[1])
    else:
        runs = run_subsample

    if method == 'RNN_poisson':
        rnn_out = True
    else:
        rnn_out = False

    for my_neuron in runs:
        print "Running neuron " + str(my_neuron)
        X_train,X_test,y_train,y_test = bookkeeping.organize_data(all_spikes=all_spikes,my_neuron=my_neuron,
            train_test_ratio=0.9, n_wins=n_wins,winsize=winsize,
            convolve_params=convolve_params,RNN_out=rnn_out,shrink_X=shrink_X,flatten_X=flatten_X)

        real_r_train, bootstrap_hist_train, pct_score_train, real_r_test, bootstrap_hist_test, pct_score_test, train_pred, test_pred = run_funct(method, X_train, X_test, y_train, y_test)

        train.append(real_r_train)
        train_pct.append(pct_score_train)
        train_bootstrap.append(bootstrap_hist_train)
        test.append(real_r_test)
        test_pct.append(pct_score_test)
        test_bootstrap.append(bootstrap_hist_test)
        train_preds.append(train_pred)
        test_preds.append(test_pred)

    np.save('train_r2',train)
    np.save('train_bootstrap',train_bootstrap)
    np.save('train_pct',train_pct)
    np.save('test_r2',test)
    np.save('test_bootstrap',test_bootstrap)
    np.save('test_pct',test_pct)
    np.save('train_preds',train_preds)
    np.save('test_preds',test_preds)

    os.chdir(save_dir)


def load_one(input_dir='.'):
    results_dict = OrderedDict()
    os.chdir(input_dir)

    params = OrderedDict()
    with open('params.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            params[row[0]] = row[1]

    results_dict['train'] = np.load('train_r2.npy')
    results_dict['train_bootstrap'] = np.load('train_bootstrap.npy')
    results_dict['train_pct'] = np.load('train_pct.npy')
    results_dict['test'] = np.load('test_r2.npy')
    results_dict['test_bootstrap'] = np.load('test_bootstrap.npy')
    results_dict['test_pct'] = np.load('test_pct.npy')
    results_dict['train_pred'] = np.load('train_pred.npy')
    results_dict['test_pred'] = np.load('test_pred.npy')

    return results_dict,params


def run_all(all_spikes, save_dir, fname, mymodels,
    run_subsample=None,
    convolve_params=None, n_wins=10,
    winsize=1, shrink_X=None,
    include_my_neuron=False, verbose=False,
    safe=True, note=None):

    """
    Runs all of the models on all of the data, evaluates, and saves them.

    """

    if os.path.exists(save_dir+fname):
        if safe:
            y = raw_input("Directory already exists. Data may be overwritten. Type y to continue.")
            assert y == 'y'
    else:
        print "Creating " + save_dir+fname
        os.makedirs(save_dir+fname)

    os.chdir(save_dir+fname)

    with open('params.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date and time: ', str(time.strftime("%c"))])
        writer.writerow(['models: ', mymodels])
        writer.writerow(['neuron subsample: ', str(run_subsample)])
        writer.writerow(['convolve params: ', str(convolve_params)])
        writer.writerow(['n_wins: ', str(n_wins)])
        writer.writerow(['winsize: ', str(winsize)])
        writer.writerow(['shrink_X: ', str(shrink_X)])
        writer.writerow(['note',str(note)])


    if run_subsample is None:
        runs = xrange(all_spikes.shape[0])
    else:
        runs = run_subsample

    for i, mymodel in enumerate(mymodels):

        train = []
        train_pct = []
        train_bootstrap = []
        test = []
        test_pct = []
        test_bootstrap = []
        train_pred = []
        test_pred = []

        if verbose:
            print "running " + mymodel + " #"+ str(i)
        if verbose:
            print "creating " + mymodel + " directory in " + save_dir
        os.makedirs(mymodel)
        os.chdir('./' + mymodel)

        if mymodel == 'RNN_poisson' or mymodel == 'XGB_poisson2d':
            rnn_out = True
        else:
            rnn_out = False

        for i, my_neuron in enumerate(runs):

            print "Running neuron " + str(my_neuron) + " run number " + str(i)

            X_train,X_test,y_train,y_test = bookkeeping.organize_data(all_spikes=all_spikes,my_neuron=my_neuron,
                subsample=None,train_test_ratio=0.9, n_wins=n_wins,winsize=winsize, flatten_X=flatten_X,
                convolve_params=convolve_params,RNN_out=rnn_out,shrink_X=shrink_X)


            real_r_train, bootstrap_hist_train, pct_score_train, real_r_test, bootstrap_hist_test, pct_score_test, this_train_pred, this_test_pred = run_funct(mymodel, X_train, X_test, y_train, y_test)
            train.append(real_r_train)
            train_pct.append(pct_score_train)
            train_bootstrap.append(bootstrap_hist_train)
            test.append(real_r_test)
            test_pct.append(pct_score_test)
            test_bootstrap.append(bootstrap_hist_test)
            train_pred.append(this_train_pred)
            test_pred.append(this_test_pred)

        np.save('train_r2',train)
        np.save('train_bootstrap',train_bootstrap)
        np.save('train_pct',train_pct)
        np.save('test_r2',test)
        np.save('test_bootstrap',test_bootstrap)
        np.save('test_pct',test_pct)
        np.save('train_pred',train_pred)
        np.save('test_pred',test_pred)
        os.chdir('..')


def load_all(input_dir,mymodels,verbose=True):
    results_dict = OrderedDict()
    params_dict = OrderedDict()

    os.chdir(input_dir)
    files = os.listdir('.')
    if verbose == True:
        print "files in this directory " + str(files)
    for mymodel in mymodels:
        if mymodel in files:
            try:
                os.chdir('./'+mymodel)
                results_dict[mymodel]=dict()

                results_dict[mymodel]['train'] = np.load('train_r2.npy')
                results_dict[mymodel]['train_bootstrap'] = np.load('train_bootstrap.npy')
                results_dict[mymodel]['train_pct'] = np.load('train_pct.npy')
                results_dict[mymodel]['test'] = np.load('test_r2.npy')
                results_dict[mymodel]['test_bootstrap'] = np.load('test_bootstrap.npy')
                results_dict[mymodel]['test_pct'] = np.load('test_pct.npy')
                results_dict[mymodel]['train_pred'] = np.load('train_preds.npy')
                results_dict[mymodel]['test_pred'] = np.load('test_preds.npy')
            except:
                print mymodel + " hasn't been run"
                continue
            try:
                params = OrderedDict()
                with open('params.csv', 'rb') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        params[row[0]] = row[1]

                params_dict[mymodel] = params
            except IOError:
                print "No params file"
                params_dict[mymodel] = OrderedDict()

            os.chdir('..')

    return results_dict,params_dict


