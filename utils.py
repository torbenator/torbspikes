import numpy as np
import matplotlib.pyplot as plt

def poisson_pseudoR2(y, y_hat, y_null=None, verbose=False):
    """
    Determines how well a model does at predicting outcomes compared to the mean of the training set.

    Parameters:
    ==========
    y = predicted variable
    y_hat = model
    y_null = null hypothesis, mean of predicted variable in training set.

    Returns:
    ==========
    R2 : pseudo r2 values
    """

    if y_null is None:
        y_null = np.mean(y)
    if np.ndim(y_hat) > 1:
        #extremely important line of code
        y_hat = np.squeeze(y_hat)
        if verbose == True:
            print "y_hat squeezed"
    if np.ndim(y) > 1:
        #extremely important line of code
        y = np.squeeze(y)
        if verbose == True:
            print "y_test squeezed"

    eps = np.spacing(1)
    L1 = np.sum(y*np.log(eps+y_hat) - y_hat)
    if any(np.isnan(y*np.log(eps+y_hat) - y_hat)):
        print "nan's found in L1. Using nanmean"
        L1 = np.nansum(y*np.log(eps+y_hat) - y_hat)
    L0 = np.sum(y*np.log(eps+y_null) - y_null)
    if any(np.isnan(y*np.log(eps+y_hat) - y_hat)):
        print "nan's found in L0. Using nanmean"
        L0 = np.nansum(y*np.log(eps+y_null) - y_null)
    LS = np.sum(y*np.log(eps+y) - y)
    R2 = 1-(LS-L1)/(LS-L0)

    if verbose:
        print "y " + str(y.shape)
        print "y_hat" + str(y_hat.shape)
        print "L1 "+str(L1)
        print "L0 "+str(L0)
        print "LS "+str(LS)
        print "R2 " + str(R2)

    return R2

def reduce_shrinkage(y_test,y_hat,weight_coeff=1):

    # TODO this totally doesn't work.

    m,y = np.polyfit(np.squeeze(y_hat),y_test,1)

    adjusted_y_hat = y_hat*m + y
    return adjusted_y_hat


def bootstrap_stats(y, y_hat, n_runs=100,plot_fig=False):
    """
    Compares a model's fit to a bootstrapped distribution of fits.

    Parameters:
    ==========
    model : any keras model
    X : Set to fit model to
    y : true values
    n_runs : number of bootstrap runs
    plot_fig : plots a histogram of the bootstrap scores and the real fit

    Returns:
    ==========
    pct_score : the percentile of the bootstrapped fits that the real fit falls into

    """

    # fit model and get real score
    real_val = poisson_pseudoR2(y,y_hat,verbose=False)

    # create bootstrap distribution by shuffling test set
    bootstrap_hist = []
    for x in xrange(n_runs):
        np.random.shuffle(y)
        bootstrap_hist.append(poisson_pseudoR2(y,y_hat))

    pct_score = (len(filter(lambda i: i<real_val,bootstrap_hist))/float(n_runs))*100

    if plot_fig == True:
        hist, bins = np.histogram(bootstrap_hist, bins=np.arange(-1,1,.02))
        width = 0.8 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        fig = plt.figure(figsize=(5,5))
        ax1 = fig.add_subplot(111)
        ax1.bar(center, hist, align='center', width=width,color='w')
        ax1.bar(real_val,10,width=.02,color='k')

        ax1.set_title("pct score: " + str(pct_score))
        ax1.set_xlabel("poisson pseudo R2")
        ax1.set_ylabel("count")
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_ticks_position('left')

    return real_val, bootstrap_hist, pct_score

def calc_solo_chorist():
    return