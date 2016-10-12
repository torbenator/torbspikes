import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from math import poisson_pseudoR2



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

    if inds == None:
        inds = range(all_spikes.shape(0))

    corrmat = np.corrcoef(all_spikes)
    corrmat = [corrmat[ind] for ind in inds]

    ax1.imshow(corrmat,cmap=plt.get_cmap('viridis'))

    fig.colorbar(ax1)
    ax1.set_title("Spike train correlations")
    if label:
        ax1.set_xlabel(label)




