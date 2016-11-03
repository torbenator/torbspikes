import numpy as np
import scipy.io
import bookkeeping, models, big_jobs, utils, plotting

def load_dat(indir):
    monkey_dat = scipy.io.loadmat(indir)
    monkey_dat.keys()
    all_spikes = monkey_dat['spikes']
    return all_spikes



if __name__ == "__main__":

    save_dir = '/Users/Torben/Code/torbspikes/'
    note = "individual runs because weird clash"
    all_spikes = load_dat(indir="/Users/Torben/Documents/Kording/GLMDeep/M1_Stevenson_Binned.mat")
    sorted_by_spike_count = bookkeeping.sort_spikes(all_spikes, method='sum')[186:196]
    method = 'NN_poisson'
    convolve_params = {"kernel_size":[3,5,10,15],"kernel_type":["cos", "cos","cos","cos"],"X":True,"y":False}


    big_jobs.run_method(all_spikes, method, save_dir, fname=method,
        run_subsample=sorted_by_spike_count,
        convolve_params=None, n_wins=5,
        winsize=1, shrink_X=.5,
        include_my_neuron=False, verbose=True,
        safe=False, note=None)