import numpy as np
import scipy.io
import bookkeeping, models, big_jobs, utils, plotting





if __name__ == "__main__":

    save_dir = '/Users/Torben/Code/torbspikes/subsample/'

    #method='RNN_poisson'
    my_methods = ['GLM_poisson','GLM_poisson2','GLM_poisson3']
    note = "individual runs because weird clash"
    all_spikes = load_dat(indir="/Users/Torben/Documents/Kording/GLMDeep/M1_Stevenson_Binned.mat")
    sorted_by_spike_count = bookkeeping.sort_spikes(all_spikes, method='sum')[186:196]

    convolve_params = {"kernel_size":[3,5,10,15],"kernel_type":["cos", "cos","cos","cos"],"X":True,"y":False}

    """
    big_jobs.run_method(all_spikes, 'RNN_poisson', save_dir='./',fname=fname ,
    run_subsample=sorted_by_spike_count,
    convolve_params=convolve_params, n_wins=5,
    winsize=1, shrink_X=0.9,
    include_my_neuron=False, verbose=False,
    safe=False, note=None)

    results_dict,params = big_jobs.load_one(input_dir='.')

    print results_dict['train'], results_dict['test']
    """
    for my_method in my_methods:
        big_jobs.run_method(all_spikes, my_method, save_dir, my_method,
            run_subsample=sorted_by_spike_count,
            convolve_params=None, n_wins=5,
            winsize=1, shrink_X=.5,
            include_my_neuron=False, verbose=True,
            safe=False, note=None)


    # d,params = big_jobs.load_all('/Users/Torben/Code/torbspikes/test_run_all/',['NN_poisson','GLM_poisson'])
    # print d['GLM_poisson']['train']
    # print params

