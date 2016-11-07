import numpy as np
import scipy.io
import bookkeeping, models, big_jobs, utils, plotting

if __name__ == "__main__":

    save_dir = '/Users/Torben/Code/torbspikes/1_ms_subsampled_10ms/'

    method='NN_poisson'
    note = "10ms_flattened_including_PCA_and_avg_now_w_self"
    all_spikes = bookkeeping.load_dat(indir="/Users/Torben/Documents/Kording/GLMDeep/M1_Stevenson_binned_1ms.mat")
    #kernel_params = {"kernel_size":[5,10,15],"kernel_type":["cos","cos","cos"],"X":True,"y":False}
    all_spikes = all_spikes[:,:20000] #subsample to speed up 10X

    sorted_by_spike_count,_ = bookkeeping.sort_spikes(all_spikes, method='sum')

    big_jobs.run_method(all_spikes, method, save_dir, method,
        run_subsample=sorted_by_spike_count[50:],
        convolve_params=None, n_wins=10,
        winsize=1, shrink_X=.5,flatten_X=True,
        include_my_neuron=True, verbose=True,
        safe=True, note=note)