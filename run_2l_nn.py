import numpy as np
import scipy.io
import bookkeeping, models, big_jobs, utils, plotting

if __name__ == "__main__":

    save_dir = '/Users/Torben/Code/torbspikes/preceeding_50ms_to_10ms_flattened/'

    method='NN_poisson_2l'
    note = "1 50 ms window predicting spiking 1 10ms bin after"
    spikes_1ms = bookkeeping.load_dat(indir="/Users/Torben/Documents/Kording/GLMDeep/4Pascal.mat")
    winsize=10
    all_spikes = bookkeeping.resample_data(spikes_1ms[:,:670500],winsize)

    #all_spikes = bookkeeping.load_dat(indir="/Users/Torben/Documents/Kording/GLMDeep/M1_Stevenson_Binned.mat")
    # sorted_spikes = bookkeeping.sort_spikes(all_spikes,'sum')
    # certain_neurons= sorted_spikes[0][100:120]


    big_jobs.run_method(all_spikes, method, save_dir, method,
        run_subsample=None,
        convolve_params=None, n_wins=5,
        shrink_X=.9,window_mean=False,
        include_my_neuron=True, verbose=False,
        safe=True, note=note)