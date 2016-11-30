import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.stats import pearsonr, sem
from utils import poisson_pseudoR2
from scipy.signal import medfilt


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



def plot_from_results_dict(results_dict, certain_neurons=None,
    verbose=False,just_test=False,plot_all=True,
    filt_bootstrap=None,plot_null=False):

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)

    model_data = []
    finite_inds = []
    labels = []

    if plot_null == True:
        labels.append("Null")
        finite_inds.append(range(100))
        model_data.append([0 for _ in range(100)])

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


def compare_methods(dat, method1, method2, color_feature,feature_name=None):

    inds1 = [i for i,j in enumerate(dat[method1]['test']) if j>0]
    inds2 = [i for i,j in enumerate(dat[method2]['test']) if j>0]
    shared_inds = np.intersect1d(inds1,inds2)

    colors = plt.cm.viridis(np.linspace(0, 1, len(color_feature)))
    cinds = np.argsort(color_feature)

    m = np.max([np.max(dat[method2]['test'][shared_inds]),np.max(dat[method1]['test'][shared_inds])])

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)

    ax1.plot([0,m],[0,m],'k',alpha=0.5)
    for i in shared_inds:
        ax1.plot(dat[method1]['test'][i],dat[method2]['test'][i],'.',c=colors[cinds[i]])
    ax1.set_xlabel(method1)
    ax1.set_ylabel(method2)
    ax1.set_xlim([0,m])
    ax1.set_ylim([0,m])

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_title("Poisson r2 of predictable neurons")

    ax2 = fig.add_axes([.6, .1, 0.05, .8])
    norm = mpl.colors.Normalize(vmin=min(color_feature), vmax=max(color_feature))
    bounds = np.linspace(min(color_feature),max(color_feature),5)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=mpl.cm.viridis,
        norm=norm, ticks=bounds, orientation='vertical')

    cb1.set_label(feature_name)


def compare_methods2(dat, plot_feature,feature_label=None):

    shared_inds = []

    colors = plt.cm.viridis(np.linspace(0, 1, len(dat.keys())))
    cind = 0
    for method in dat.keys():

        inds = [i for i,j in enumerate(dat[method]['test']) if j>0]
        shared_inds = np.intersect1d(shared_inds,inds)

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)
    for method in dat.keys():

        ax1.plot(plot_feature, dat[method]['test'], marker='.',linewidth=0,color=colors[cind],alpha=0.5,markersize=2)

        sorted_by_feature = [(x,y) for (y,x) in sorted(zip(plot_feature,dat[method]['test']))]
        vals, inds = zip(*sorted_by_feature)
        ax1.plot(inds, medfilt(vals,11),color=colors[cind],label=method)

        cind+=1

    ax1.set_xlabel(feature_label)
    ax1.set_ylabel("Pseudo r2")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_title("Poisson r2 of predictable neurons")
    ax1.legend(loc='best')


def compare_methods3(dat, plot_feature,feature_label=None):

    shared_inds = []

    colors = plt.cm.viridis(np.linspace(0, 1, len(dat.keys())+1))
    for method in dat.keys():
        inds = [i for i,j in enumerate(dat[method]['test']) if j>0]
        shared_inds = np.intersect1d(shared_inds,inds)

    bins = [j for i,j in enumerate(np.sort(plot_feature)) if np.mod(i,10) == 0]
    bin_inds = range(len(bins)-1)
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ci=0
    mi=0
    for method in dat.keys():
        labelme=True
        for bi in bin_inds:
            model_data = [j for i,j in zip(plot_feature,dat[method]['test']) if i>= bins[bi] and i<bins[bi+1] and j>0]
            ax1.errorbar(bi+mi, np.mean(model_data), yerr=sem(model_data), ecolor=colors[ci],linestyle="None",elinewidth=3,label=method if labelme==True else None)
            labelme=False
        ci+=1
        mi+=0.2

    ax1.set_xlabel(feature_label)
    ax1.set_ylabel("Pseudo r2")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xticks(bin_inds)
    ax1.set_xticklabels(bins[:-1])
    ax1.set_title("Poisson r2 of predictable neurons")
    ax1.legend(loc='best')


def compare_methods4(dat,method1,method2,plot_feature,feature_label=None,logy=False):

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Relationship between " + feature_label + " and differences in predictability")

    if logy == True:
        ax1.semilogy(dat[method1]['test'] - dat[method2]['test'],plot_feature,'.')
    else:
        ax1.plot(dat[method1]['test'] - dat[method2]['test'],plot_feature,'.')

    ax1.set_xlabel(method1 + ' - ' + method2)
    ax1.set_ylabel(feature_label)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    finite_matches = [(i,j,k) for i,j,k in zip(dat[method1]['test'],dat[method2]['test'],plot_feature) if np.isfinite(i) and np.isfinite(j)]
    unzipped = zip(*finite_matches)

    print pearsonr([i-j for i,j in zip(unzipped[0],unzipped[1])],unzipped[2])


def total_predicted(dat):

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    pcts = []
    means = []
    labels = []
    xticks = range(len(dat.keys()))

    for method in dat.keys():
        this_dat = filter(lambda x: x>0,dat[method]['test'])
        pcts.append(len(this_dat)/float(len(dat[method]['test'])))
        means.append(np.mean(this_dat))
        labels.append(method)
    ax1.bar(xticks,pcts,align='center')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.set_xticks(xticks)
    ax1.set_title("Percent of neurons that could be predicted")
    ax1.set_xticklabels([i for i in dat.keys()],rotation=90)

    ax2 = fig.add_subplot(122)

    ax2.bar(xticks,means,align='center')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.set_xticks(xticks)
    ax2.set_title("Average pseudo-r2 of Prediction")
    ax2.set_xticklabels([i for i in dat.keys()],rotation=90)



def _get_weights(dat,my_neuron=None):
    if my_neuron is not None:
        glm_weights = dat['GLM_poisson']['weights'][my_neuron][0]
        xgb_weights = dat['XGB_poisson']['weights'][my_neuron]
    else:
        glm_weights = []
        xgb_weights = []
        for neuron in range(len(dat['GLM_poisson']['test'])):
            glm_weights.extend(np.abs(np.squeeze(dat['GLM_poisson']['weights'][neuron][0])))
            xgb_weights.extend(np.squeeze(dat['XGB_poisson']['weights'][neuron]))
    return glm_weights,xgb_weights


def visualize_weights(dat,my_neuron=None,log_hist=False):

    g,x = _get_weights(dat,my_neuron)

    nullfmt = NullFormatter()

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]


    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)


    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(g, x,alpha=0.5,lw = 0,s=3)

    # now determine nice limits by hand:
    xymax = np.max([np.max(np.fabs(g)), np.max(np.fabs(x))])
    #lim = (int(xymax/binwidth) + 1) * binwidth

    glm_range = [0,max(g)]
    xgb_range = [0,max(x)]
    axScatter.set_xlim(glm_range)
    axScatter.set_ylim(xgb_range)

    glm_bin_width = 0.01
    xgb_bin_width = 5
    bins1 = np.arange(glm_range[0], glm_range[1] + glm_bin_width, glm_bin_width)
    bins2 = np.arange(xgb_range[0], xgb_range[1] + xgb_bin_width, xgb_bin_width)
    axHistx.hist(g, bins=bins1,log=log_hist)
    axHisty.hist(x, bins=bins2, orientation='horizontal',log=log_hist)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())



    axScatter.spines['right'].set_visible(False)
    axScatter.spines['top'].set_visible(False)
    axScatter.xaxis.set_ticks_position('bottom')
    axScatter.yaxis.set_ticks_position('left')
    axScatter.set_xlabel('GLM weights')
    axScatter.set_ylabel('XGB weights')

    axHistx.spines['right'].set_visible(False)
    axHistx.spines['top'].set_visible(False)
    axHistx.xaxis.set_ticks_position('bottom')
    axHistx.yaxis.set_ticks_position('left')

    axHisty.spines['right'].set_visible(False)
    axHisty.spines['top'].set_visible(False)
    axHisty.xaxis.set_ticks_position('bottom')
    axHisty.yaxis.set_ticks_position('left')
    plt.show()

