import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import emd
from scipy import ndimage
import ghostipy as gsp
import gc
import ssqueezepy as sq

from sigprocess import *
cma = np.array([-4.4019e-004	, 1.2908e-003,	-1.9633e-002])
La = np.array([-8.3023e-019, 	-8.1e-002,	-8.835e-002])
posa = La-cma

cmb = np.array([8.0563e-005,	5.983e-004,	-6.8188e-003])
Lb = np.array([5.3302e-018, -7.233e-002, 3.12e-002+2.0e-003])
posb = Lb-cmb

num = 9
df = pd.read_csv('DATA/A/data_{}.csv'.format(num), index_col='t')

df['cur'] = df['cur']*0.0005
df['rot'] = np.deg2rad(df['rot'])
# PSD(df,1660)
data, t, fs, dt = prep_data(df, 1660, 415, 10)
# PSD(data,fs)

C = pd.DataFrame({'cur': data[:,1],'rot': data[:,0]},t)
C.plot()
B = imu2body(data[:,8:], t, fs, posb)
B[['thx','thy','thz']].plot()
B[['Ax','Ay','Az']].plot()

A = imu2body(data[:,2:8],t, fs, posa)
 




_b, _t = scipy.signal.resample(B, len(B)//10, t=t, axis=0, window='hann')
b = pd.DataFrame(_b,_t, columns=B.columns)

# B.insert(0, 'rot', data[:,0])
# B.insert(0, 'cur', data[:,1])
PSD(B, fs)
PSD(A, fs)
kwargs_dict = {}
kwargs_dict['cmap'] = plt.cm.Spectral_r
kwargs_dict['vmin'] = 0
kwargs_dict['vmax'] = 1
kwargs_dict['linewidth'] = 0
kwargs_dict['rasterized'] = True
kwargs_dict['shading'] = 'gouraud'

Cur = gsp.analytic_signal(B['Dz'])

coefs_cwt, _, f_cwt, t_cwt, _ = gsp.cwt(b['Ay'].to_numpy(),fs=fs//10,timestamps=tt,boundary='mirror', freq_limits=[1, 360], voices_per_octave=16)
psd_cwt = coefs_cwt.real**2 + coefs_cwt.imag**2
psd_cwt /= np.max(psd_cwt)

fig = plt.figure()
pc_cwt = plt.pcolormesh(t_cwt, f_cwt, psd_cwt, **kwargs_dict)


coefs_wsst, _, f_wsst, t_wsst, _  = gsp.wsst(b['Ay'].to_numpy(),fs=fs//10,timestamps=tt, freq_limits=[1, 360], voices_per_octave=16)
psd_wsst = coefs_wsst.real**2 + coefs_wsst.imag**2
psd_wsst /= np.max(psd_wsst)



fig = plt.figure()
pc_wsst = plt.pcolormesh(t_wsst, f_wsst, psd_wsst, **kwargs_dict)
pc_wsst.set_edgecolor('face')
cbar_wsst = fig.colorbar(pc_wsst)
cbar_wsst.ax.tick_params(labelsize=16)
cbar_wsst.set_label("Normalized PSD", fontsize=16, labelpad=15)

# %%
S = b[['Ay']].to_numpy()
mfreqs = np.array([360,300,240,180,120,90,60,30,15,7.5])
imf, _ = emd.sift.mask_sift(S, mask_freqs=mfreqs/fs//10,  mask_amp_mode='ratio_sig', ret_mask_freq=True, nphases=8, mask_amp=5)
Ip, If, Ia = emd.spectra.frequency_transform(imf, fs, 'nht')
emd.plotting.plot_imfs(imf,tt, scale_y=True, cmap=True)
emd.plotting.plot_imfs(Ia,tt, scale_y=True, cmap=True)
emd.plotting.plot_imfs(Ip,tt, scale_y=True, cmap=True)
emd.plotting.plot_imfs(If,tt, scale_y=True, cmap=True)

config = emd.sift.get_config('mask_sift')
config['mask_amp_mode'] = 'ratio_sig'
config['mask_amp'] = 2
config['max_imfs'] = 5
config['imf_opts/sd_thresh'] = 0.05
config['envelope_opts/interp_method'] = 'mono_pchip'
imf2 = emd.sift.mask_sift_second_layer(IA, mask_freqs, sift_args=config)
IP2, IF2, IA2 = emd.spectra.frequency_transform(imf2, fs, 'nht')
spec = emd.spectra.hilberthuang_1d(If, Ia, freq_edges)

freq_edges, freq_bins = emd.spectra.define_hist_bins(1, 415, 414, 'log')
hht = emd.spectra.hilberthuang(If, Ia, freq_edges, mode='energy')
shht = ndimage.gaussian_filter(hht, 2)
fig = plt.figure(figsize=(10, 6)) 
emd.plotting.plot_hilberthuang(shht, t, freq_bins, fig=fig,freq_lims=(1, 415), log_y=True, cmap=plt.cm.Spectral_r)




# %%


