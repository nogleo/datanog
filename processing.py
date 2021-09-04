import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import emd
from scipy import ndimage

from sigprocess import *
cm = np.array([8.0563e-005,	5.983e-004,	-6.8188e-003])
L = np.array([5.3302e-018, -7.233e-002, 3.12e-002+2.0e-003])
pos = L-cm

num = 0
df = pd.read_csv('DATA/A/data_{}.csv'.format(num), index_col='t')

data, t, fs, dt = prep_data(df, 1660, 415, 5)


# A = imu2body(data[:,2:8],t, fs)
B = imu2body(data[:,8:], t, fs, pos)
B.insert(0, 'rot', data[:,0])
B.insert(0, 'cur', data[:,1]/10000)

PSD(B, fs)
# %%
S = B[['Az']].to_numpy()

imf, mask_freqs = emd.sift.mask_sift(S, mask_freqs=240/fs,  mask_amp_mode='ratio_sig', ret_mask_freq=True, nphases=8, mask_amp=5, mask_step_factor=2)

imf, noise = emd.sift.complete_ensemble_sift(S,nensembles=5, nprocesses=10)
mfreqs = mask_freqs*fs
IP, IF, IA = emd.spectra.frequency_transform(imf, fs, 'nht')
emd.plotting.plot_imfs(imf,t, scale_y=True, cmap=True)
emd.plotting.plot_imfs(IA,t, scale_y=True, cmap=True)
emd.plotting.plot_imfs(IP,t, scale_y=True, cmap=True)
emd.plotting.plot_imfs(IF,t, scale_y=False, cmap=True)

config = emd.sift.get_config('mask_sift')
config['mask_amp_mode'] = 'ratio_sig'
config['mask_amp'] = 2
config['max_imfs'] = 5
config['imf_opts/sd_thresh'] = 0.05
config['envelope_opts/interp_method'] = 'mono_pchip'
imf2 = emd.sift.mask_sift_second_layer(IA, mask_freqs, sift_args=config)
IP2, IF2, IA2 = emd.spectra.frequency_transform(imf2, fs, 'nht')
spec = emd.spectra.hilberthuang_1d(IF, IA, freq_edges)

emd.plotting.plot



freq_edges, freq_bins = emd.spectra.define_hist_bins(1, 300, 299, 'log')
freq_edges2, freq_bins2 = emd.spectra.define_hist_bins(1, 300, 299, 'log')
hht = emd.spectra.hilberthuang(IF, IA, freq_edges, mode='amplitude')
holo = emd.spectra.holospectrum(IF,IF2, IA2, freq_edges, freq_edges2)
shht = ndimage.gaussian_filter(hht, 1)
fig = plt.figure(figsize=(10, 6)) 
emd.plotting.plot_hilberthuang(shht, t, freq_bins, fig=fig,freq_lims=(2, 300), log_y=True, cmap='inferno')
fig2 = plt.figure(figsize=(10, 6))
emd.plotting.plot_holospectrum(hht,freq_bins, freq_bins2,  fig=fig2,freq_lims=(2, 300))



# %%
H = signal.hilbert(B[['Dx','Dy','Dz', 'thx','thy','thz']], axis=0)
Ia = np.abs(H)


