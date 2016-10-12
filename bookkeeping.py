import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict,defaultdict

"""
Functions to organize and manipulate spiking data

"""


def _build_kernel(kernel_size,kernel_type="cos"):
    """
    Builds different kinds of kernals to use for smoothing spike trains.
    Kernel types:
    cos : 1/2 cosine where
    """
    if kernel_type == "cos":
        return (cos(np.linspace(np.pi,3*np.pi,kernel_size))+1)*0.5

    if kernel_type == "exp":
        exp = np.linspace(0,10,kernel_size)**2
        return exp/float(max(exp))
    if kernel_type == "shifted_cos":
        return np.concatenate(((cos(np.linspace(np.pi,3*np.pi,kernel_size))+1)*0.5, np.zeros(kernel_size)))
    else:
        print kernel_type + " not built yet"


def organize_data(all_spikes,my_neuron=193,subsample=100,
                  train_test_ratio=0.9,winsize=None,
                  convolve_params=None,RNN_out=False,
                  verbose=False, include_my_neuron=False,
                  shrink_X=None, to_binary=False):

    """

    Monstrosity of a bookkeeping method that builds train and test sets for predicting a
    spike train using nearby spike trains

    Parameters
    ============
    all_spikes : np.array of n_spikes x bins
    my_neuron : index of spike train we're trying to guess
    subsample : number of neurons to use as parameters set to 0 or all_spikes.shape[1]
    to include all spiketrains except for the one we're guessing
    train_test_ratio : self explanatory
    convolve_params : dictionary of kernel sizes and types to convolve with data
    as well as whether to convolve them with features or predictor(optional)
    EXAMPLE: kernel_params = {"kernel_size":[5,10,15],"kernel_type":["cos","cos","cos"],"X":True,"y":False}
    winsize : window_size to use if using only information preceeding spikes
    RNN_out : (boolean, requires winsize) True if you want X to be (Features x Examples),
    False if you want X to be (Features x Example_d1 x Example_d2) --> needed for RNN so that each example
    can have a feature and time component
    shrink_X : multiplies the spike series by the integer you input to normalize spikes - helps with NN algorithms
    to_binary : converts spike counts into simple 1/0 spikes/no spikes matrix

    Returns
    ===========
    X_train,X_test,y_train,y_test

    """

    if convolve_params == None:
        convolve_params = {"kernel_size":None,"X":False,"y":False}

    # run on a subsample of the data or not
    if subsample > 0 and subsample < all_spikes.shape[0]:
        these_neurons = np.random.choice(all_spikes.shape[0],subsample,replace=False)
    else:
        these_neurons = range(all_spikes.shape[0])

    # should we include the neuron we're trying to guess in the train set?
    these_neurons = [i for i in these_neurons if i != my_neuron or include_my_neuron==True]
    if verbose:
        print "Using "+str(len(these_neurons))+" X neurons:"

    # building kernels to convolve spike train with
    if convolve_params["kernel_size"]:
        new_y_inds = [] # exclude these from feature matrix
        processed_dat = []
        for kernel_set in zip(kernel_params['kernel_size'],kernel_params['kernel_type']):
            kernel = _build_kernel(kernel_size=kernel_set[0],kernel_type=kernel_set[1])
            for i, row in enumerate(all_spikes):
                conv = np.convolve(row,kernel,'same')
                normalized_conv = conv/float(max(conv))
                if sum(np.isnan(normalized_conv)) > 0:
                    if verbose:
                        print "excluding spike train " + str(i) + ". Kernel " + str(kernel_set) + " created nans."
                else:
                    processed_dat.append(normalized_conv) # normalizing. big filters have big amps
                    if i == my_neuron:
                        new_y_inds.append(len(processed_dat)-1)
        processed_dat = np.array(processed_dat)
        these_neurons = [i for i in xrange(processed_dat.shape[0]) if i not in new_y_inds]

        if verbose:
            print "processed_dat shape: " + str(processed_dat.shape)
            print "excluded " + str(len(new_y_inds)) + " rows of processed matrix. " + str(len(these_neurons)) + " features to be used"


    # setting train and test inds
    # for training using only preceeding information
    if winsize:
        print "using only data preceeding spikes with window_size: "+ str(winsize)
        split_ind = int((all_spikes.shape[1]-winsize)*train_test_ratio)

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        if convolve_params["X"] == True:
            for d in xrange(split_ind):
                X_train.append([processed_dat[n,[i for i in range(d,d+winsize)]] for n in these_neurons])
            X_train = np.array(X_train)

            for d in xrange(split_ind,processed_dat.shape[1]-winsize):
                X_test.append([processed_dat[n,[i for i in range(d,d+winsize)]] for n in these_neurons])
            X_test = np.array(X_test)
        else:
            for d in xrange(split_ind):
                X_train.append([all_spikes[n,[i for i in range(d,d+winsize)]] for n in these_neurons])
            X_train = np.array(X_train)

            for d in xrange(split_ind,all_spikes.shape[1]-winsize):
                X_test.append([all_spikes[n,[i for i in range(d,d+winsize)]] for n in these_neurons])
            X_test = np.array(X_test)

        if convolve_params["y"] == True:
            y_train = processed_dat[my_neuron,winsize:split_ind+winsize].T
            y_test = processed_dat[my_neuron,split_ind+winsize:processed_dat.shape[1]].T

        else:
            y_train = all_spikes[my_neuron,winsize:split_ind+winsize].T
            y_test = all_spikes[my_neuron,split_ind+winsize:all_spikes.shape[1]].T

        if RNN_out == False:
            X_train = np.mean(X_train,2)
            X_test = np.mean(X_test,2)
        else:
            X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[2], X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[2], X_test.shape[1]))

    # for training using any information
    else:
        train_inds = np.random.choice(all_spikes.shape[1],int(all_spikes.shape[1]*train_test_ratio),replace=False)
        test_inds = [i for i in range(all_spikes.shape[1]) if i not in train_inds]
        if verbose:
            print "length of train inds: " + str(len(train_inds))
            print "length of test inds: " + str(len(test_inds))

        # convolve X spikes?
        if convolve_params["X"] == True:
            X_train = np.array([processed_dat[i,train_inds] for i in these_neurons]).T
            X_test = np.array([processed_dat[i,test_inds] for i in these_neurons]).T
        else:
            X_train = np.array([all_spikes[i,train_inds] for i in these_neurons]).T
            X_test = np.array([all_spikes[i,test_inds] for i in these_neurons]).T

        # convolve y spikes?
        if convolve_params["y"] == True:
            y_train = processed_dat[my_neuron,train_inds].T
            y_test = processed_dat[my_neuron,test_inds].T
        else:
            y_train = all_spikes[my_neuron,train_inds].T
            y_test = all_spikes[my_neuron,test_inds].T

    if shrink_X:
        if verbose:
            print "shrinking X by " + str(shrink_X)
        X_train_z = (X_train-np.mean(X_train))/np.std(X_train)
        X_train = (X_train_z-np.min(X_train_z))*shrink_X

        X_test_z = (X_test-np.mean(X_test))/np.std(X_test)
        X_test = (X_test_z-np.min(X_test_z))*shrink_X

    if to_binary:
        X_train = np.array(X_train>0,dtype=int)
        X_test = np.array(X_test>0,dtype=int)
        y_train = np.array(y_train>0,dtype=int)
        y_test = np.array(y_test>0,dtype=int)

    return X_train,X_test,y_train,y_test


def sort_spikes(all_spikes, method):
    """

    A method to sort neurons by their spike counts.

    Parameters
    ==========
    all_spikes : neuron x spike count matrix
    method : method to sort neurons by
        'sum' : sort neurons by total spike count
        'residual_distance' : sort neurons by how much their spiking deviates from the field
        'sum_corr' : sort neurons by how they correlate with the other neurons

    Returns:
    ==========
    inds : sorted indicies based on method that was used

    """


    if method == 'sum':
        inds = np.argsort(np.nansum(all_spikes,1))

    elif method == 'residual_distance':
        mean_time_series = np.mean(all_spikes,1)
        residuals = []
        for spike in all_spikes:
            diff = np.sum((spike-mean_time_series)**2)
            residuals.append(diff)
        inds = np.argsort(residuals)

    elif method == "sum_corr":
        mat = np.corrcoef(all_spikes)
        inds = np.argsort(np.nansum(mat,1))

    else:
        print method + " not built yet."
        return

    return inds
