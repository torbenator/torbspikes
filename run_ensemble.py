import os
import numpy as np
import scipy.io
import bookkeeping, models, big_jobs, utils, plotting
import csv
import time

if __name__ == "__main__":

    safe = True
    n_wins=3
    winsize=1
    shrink_X = None

    load_dir = '/Users/Torben/Code/torbspikes/l2_3/'
    save_dir = '/Users/Torben/Code/torbspikes/l2_3/'

    methods =['GLM_poisson','NN_poisson','RNN_poisson','XGB_poisson']
    note = None #"l1=0.01"
    all_spikes = bookkeeping.load_dat(indir="/Users/Torben/Documents/Kording/GLMDeep/M1_Stevenson_Binned.mat")

    convolve_params = {"kernel_size":[3,5,10,15],"kernel_type":["cos", "cos","cos","cos"],"X":True,"y":False}
    runs,counts = sorted_by_spike_count = bookkeeping.sort_spikes(all_spikes, method='sum')

    runs = runs[50:]
    ensemble_train_preds = []
    ensemble_test_preds = []
    for method in methods:
        if method in os.listdir(load_dir):
            os.chdir(load_dir+method)
            print "loaded " + method + " shape: " + str(np.load('train_preds.npy').shape)
            ensemble_train_preds.append(np.load('train_preds.npy'))
            ensemble_test_preds.append(np.load('test_preds.npy'))
    all_model_train = np.squeeze(np.array(ensemble_train_preds)) # model x neuron x features
    all_model_test = np.squeeze(np.array(ensemble_test_preds)) # model x neuron x features
    print all_model_train.shape

    train = []
    train_pct = []
    train_bootstrap = []
    test = []
    test_pct = []
    test_bootstrap = []
    train_preds = []
    test_preds = []

    for i, my_neuron in enumerate(runs):
        this_neuron_train = all_model_train[:,i,:].T
        this_neuron_test = all_model_test[:,i,:].T

        print "Running neuron " + str(my_neuron)

        _,_,y_train,y_test = bookkeeping.organize_data(all_spikes=all_spikes,my_neuron=my_neuron,
            subsample=None,train_test_ratio=0.9, n_wins=n_wins,winsize=winsize,
            convolve_params=None,RNN_out=False,shrink_X=None)

        real_r_train, bootstrap_hist_train, pct_score_train, real_r_test, bootstrap_hist_test, pct_score_test, Yr, Yt = big_jobs.run_funct('XGB_ensemble', this_neuron_train, this_neuron_test, y_train, y_test)

        train.append(real_r_train)
        train_pct.append(pct_score_train)
        train_bootstrap.append(bootstrap_hist_train)
        test.append(real_r_test)
        test_pct.append(pct_score_test)
        test_bootstrap.append(bootstrap_hist_test)
        train_preds.append(Yr)
        test_preds.append(Yt)

    if 'XGB_ensemble' not in os.listdir(save_dir):
        os.makedirs(save_dir + 'XGB_ensemble')
    else:
        if safe:
            y = raw_input("Directory already exists. Data may be overwritten. Type y to continue.")
            assert y == 'y'
    os.chdir(save_dir+'XGB_ensemble')
    np.save('train_r2',train)
    np.save('train_bootstrap',train_bootstrap)
    np.save('train_pct',train_pct)
    np.save('test_r2',test)
    np.save('test_bootstrap',test_bootstrap)
    np.save('test_pct',test_pct)
    np.save('train_preds',train_preds)
    np.save('test_preds',test_preds)


    with open('params.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date and time: ', str(time.strftime("%c"))])
        writer.writerow(['method', 'ensemble xgb'])
        writer.writerow(['neuron subsample: ', str(runs)])
        writer.writerow(['convolve params: ', convolve_params])
        writer.writerow(['n_wins: ', str(n_wins)])
        writer.writerow(['winsize: ', str(winsize)])
        writer.writerow(['shrink_X: ', str(shrink_X)])
        writer.writerow(['note',str(note)])





