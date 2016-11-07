import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, sem
from utils import poisson_pseudoR2
from scipy.stats import sem


def plot_preds(y_test,y_hat,neuron_n=None,plot_mean=True):
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)

    if np.ndim(y_hat)>1:
        y_hat = np.squeeze(y_hat)

    #noise to make plot look nice
    y_noise = np.random.randn(np.size(y_test))
    nnoise = 0.1
    ax1.scatter(y_test+nnoise*y_noise,y_hat, color='b')
    if plot_mean:
        ax1.plot([min(y_test),max(y_test)],[np.mean(y_test),np.mean(y_test)],'r-',label='y_test mean',linewidth=2)
    ax1.set_title("Predicting "+ str(neuron_n) +". r = " +str(pearsonr(y_test,y_hat)))
    ax1.set_xlabel("test set")
    ax1.set_ylabel("y hat")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.legend(loc=2)


def plot_residuals(y_test,y_hat,neuron_n=None,plot_mean=True):

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)

    if np.ndim(y_hat)>1:
        y_hat = np.squeeze(y_hat)

    #noise to make plot look nice
    ax1.scatter(xrange(len(y_hat)),y_hat-y_test, color='b',label="Residuals of prediction")
    if plot_mean:
        ax1.scatter(xrange(len(y_hat)),y_hat-np.mean(y_test), color='r',label="Residuals of the mean")
        ax1.plot([0,len(y_hat)],[np.mean(y_hat-np.mean(y_test)),np.mean(y_hat-np.mean(y_test))], c='r',linewidth=2)

    ax1.plot([0,len(y_hat)],[np.mean(y_hat-y_test),np.mean(y_hat-y_test)], c='b',linewidth=2)

    ax1.set_title("Predicting neuron "+ str(neuron_n) +". Pseudo r2 = " +str(poisson_pseudoR2(y_test,y_hat)))
    ax1.set_xlabel("test set")
    ax1.set_xlim([0,len(y_hat)])
    ax1.set_ylabel("y hat")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.legend(loc=2)

def plot_consistancy_mat(all_spikes,inds=None,label=None):

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)

    if inds is None:
        inds = range(all_spikes.shape(0))

    corrmat = np.corrcoef(all_spikes)
    corrmat = [corrmat[ind] for ind in inds]

    ax1.imshow(corrmat,cmap=plt.get_cmap('viridis'))

    fig.colorbar(ax1)
    ax1.set_title("Spike train correlations")
    if label:
        ax1.set_xlabel(label)


def plot_model_r2_vals(output_arrays, labels=None,colors=None):

    means = [np.mean(i) for i in output_arrays]
    sems = [sem(i) for i in output_arrays]
    if colors is None:
        colors = ['b' for i in range(len(output_arrays))]
    if labels is  None:
        labels = range(len(output_arrays))

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)
    for i in range(len(output_arrays)):
        ax1.errorbar(i, means[i], yerr=sems[i], ecolor=colors[i],linestyle="None",elinewidth=3)
        ax1.scatter(i, means[i], marker='s', color=colors[i],lw = 0,s=50)
    ax1.set_xticks(range(len(output_arrays)))
    ax1.set_xticklabels(labels,rotation=90)
    ax1.set_xlabel("model")
    ax1.set_ylabel("r2 value")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')



def plot_from_results_dict(results_dict, certain_neurons=None, verbose=False,just_test=False,plot_all=True,filt_bootstrap=None):

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)

    model_data = []
    finite_inds = []
    labels = []

    for model in results_dict.keys():
        for k in results_dict[model].keys():
            if k == 'test' or (k == 'train' and just_test == False):
                if filt_bootstrap is not None:
                    fi = [i for i,j in enumerate(results_dict[model][k+"_pct"]) if j>filt_bootstrap]
                else:
                    fi = [i for i,j in enumerate(results_dict[model][k]) if np.isfinite(j)]

                if certain_neurons is not None:
                    fi = np.intersect1d(certain_neurons,fi)
                    if verbose:
                        print str(np.setdiff1d(certain_neurons,fi)) + " are not finite in " + k

                labels.append(model+' '+k)
                finite_inds.append(fi)
                model_data.append([results_dict[model][k][i] for i in fi])

    if just_test == True:
        colors = plt.cm.viridis(np.linspace(0, 1, len(finite_inds)))

    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(finite_inds)/2))

    cind = 0

    for i in xrange(len(finite_inds)):
        ax1.errorbar(i, np.mean(model_data[i]), yerr=sem(model_data[i]), ecolor=colors[cind],linestyle="None",elinewidth=3)
        ax1.scatter(i, np.mean(model_data[i]), marker='s', color=colors[cind],lw = 0,s=30)
        if verbose:
            print "mean of "+ str(labels[i]) + ": "+str(np.mean(model_data[i]))

        if plot_all == True:
            jitter = np.random.randn(np.size(model_data[i]))
            jitter_amp = 0.05
            ax1.scatter((np.ones(len(model_data[i]))*i)+jitter_amp*jitter, model_data[i], marker='.', color=colors[cind],lw = 0,s=3)

        if just_test == True:
            cind+=1

        elif np.mod(i,2) == 1:
            cind+=1

    ax1.set_xticks(xrange(len(finite_inds)))
    ax1.set_xticklabels(labels,rotation=90)
    ax1.set_xlabel("model")
    ax1.set_ylabel("r2 value")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.legend(loc=2)



def plot_results_dict_by_feature(results_dict, feature_array, feature_name=None, just_test=True,):

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)

    #inds = np.argsort(np.sum(all_spikes,1))

    model_data = []
    finite_inds = []
    labels = []

    for model in results_dict.keys():
        for k in results_dict[model].keys():
            if k == 'test' or (k == 'train' and just_test == False):
                print model + " " + k + " " + str(pearsonr(feature_array,results_dict[model][k]))
                plt.plot(feature_array,results_dict[model][k],label=model)

    ax1.set_xticks(xrange(len(finite_inds)))
    ax1.set_xticklabels(labels,rotation=90)
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel("r2 value")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.legend(loc=2)






