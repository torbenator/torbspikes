import numpy as np
import scipy.io
import bookkeeping, models, big_jobs, utils, plotting

if __name__ == "__main__":

    save_dir = '/Users/Torben/Code/torbspikes/1ms_binned_5wins/'

    method='RNN_poisson'
    note = "L2reg0.3_5ms_wins"
    all_spikes = bookkeeping.load_dat(indir="/Users/Torben/Documents/Kording/GLMDeep/M1_Stevenson_binned_1ms.mat")
    all_spikes = all_spikes[:,:20000] #subsample to speed up 10X

    sorted_by_spike_count = bookkeeping.sort_spikes(all_spikes, method='sum')[50:]

    big_jobs.run_method(all_spikes, method, save_dir, method,
        run_subsample=sorted_by_spike_count,
        convolve_params=None, n_wins=5,
        winsize=1, shrink_X=.5,flatten_X=False,
        include_my_neuron=False, verbose=True,
        RNN_out=False, safe=True, note=note)