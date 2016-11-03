import numpy as np
import scipy.io
import bookkeeping, models, big_jobs, utils, plotting

if __name__ == "__main__":

    save_dir = '/Users/Torben/Code/torbspikes/5_win_include_self/'

    method='NN_poisson_2l'
    note = None#"l1=0.1 flattened"
    all_spikes = bookkeeping.load_dat(indir="/Users/Torben/Documents/Kording/GLMDeep/M1_Stevenson_Binned.mat")
    sorted_by_spike_count = bookkeeping.sort_spikes(all_spikes, method='sum')[50:]

    convolve_params = {"kernel_size":[3,5,10,15],"kernel_type":["cos", "cos","cos","cos"],"X":True,"y":False}

    big_jobs.run_method(all_spikes, method, save_dir, method,
        run_subsample=sorted_by_spike_count,
        convolve_params=None, n_wins=5,
        winsize=1, shrink_X=.5,flatten_X=False,
        include_my_neuron=True, verbose=True,
        safe=True, note=note)