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
import os

from sigprocess import *



root = os.getcwd()
# %%





for file in os.listdir('DATA'):
    num = int(file[5:-4])
    print(num)

    df = pd.read_csv('DATA/data_{}.csv'.format(num), index_col='t')
    A, B, C, FS = prep_data(df, 1660, 480, 10)
    try:
        os.mkdir('./PROCDATA/data_{}'.format(num))
        os.chdir('./PROCDATA/data_{}'.format(num))
    except:
        os.chdir('./PROCDATA/data_{}'.format(num))
    
    A.to_csv('A.csv')
    B.to_csv('B.csv')
    C.to_csv('C.csv')
    os.chdir(root)
# PSD(df,1660)

# PSD(A,FS)
# PSD(B,FS)
# PSD(C,FS)

# %%
num = 8
A = pd.read_csv('./PROCDATA/data_{}/A.csv'.format(num), index_col=0)
B = pd.read_csv('./PROCDATA/data_{}/B.csv'.format(num), index_col=0)
C = pd.read_csv('./PROCDATA/data_{}/C.csv'.format(num), index_col=0)


PSD(B, fs)
PSD(B[['Ax', 'Ay', 'Az']], fs)
PSD(np.unwrap(C.rot), fs)


# %%
df=C
frames = df.columns
S = df.to_numpy()
t = df.index

spect(df,fs)

apply_emd(df, fs)
WSST(df, fs)




c_cwt, _, f_cwt, t_cwt, _ = gsp.cwt(S, fs=fs, voices_per_octave=32, method='ola', boundary='periodic', freq_limits=[1,480])


coefs_cwt, _, f_cwt, t_cwt, _ = gsp.cwt(S,timestamps=t,fs=fs, boundary='zeros', method='ola', voices_per_octave=32)
psd_cwt = c_cwt.real**2 + c_cwt.imag**2
psd_cwt /= np.max(psd_cwt)
plt.figure()
ax = plt.yscale('linear')
plt.imshow(psd_cwt, aspect='auto', cmap='turbo',extent=[tt[0], tt[-1], ff[0], ff[-1]])
ax.set_xlabel('time')
vizspect(t_cwt, ff, psd_cwt, frame,  fscale='log')
plt.figure()
Pxx = 10*np.log10(np.abs(c_cwt))
plt.pcolormesh(t_cwt, f_cwt, Pxx, **kwargs_dict)

# %%
coefs_wsst, _, f_wsst, t_wsst, _  = gsp.wsst(df.to_numpy(),fs=fs,timestamps=df.index, freq_limits=[1, 480], voices_per_octave=32, boundary='zeros', method='ola')
psd_wsst = coefs_wsst.real**2 + coefs_wsst.imag**2
vizspect(t_wsst, f_wsst[::-1], psd_wsst, frame, fscale='log')

plt.figure()
plt.imshow(psd_wsst, aspect='auto', cmap='turbo' ,extent=[t_wsst[0], t_wsst[-1], f_wsst[-1], f_wsst[0]])
# psd_wsst /= np.max(psd_wsst)
plt.show()
plt.figure()


pc_wsst = plt.pcolormesh(t_wsst, f_wsst, psd_wsst, **kwargs_dict)
pc_wsst.set_edgecolor('face')
cbar_wsst = plt.colorbar(pc_wsst)
cbar_wsst.ax.tick_params(labelsize=16)
cbar_wsst.set_label("PSD", fontsize=16, labelpad=15)


# gsp.plot_wavelet_spectrogram(coefs_wsst, f_wsst, t_wsst, **kwargs_dict, kind='power')
# %%
S = B[['Ay']].to_numpy()
t = B.index


mfreqs = np.array([360,300,240,180,120,90,60,30,15,7.5])
imf, _ = emd.sift.mask_sift(S, mask_freqs=mfreqs/fs,  mask_amp_mode='ratio_sig', ret_mask_freq=True, nphases=8, mask_amp=S.max())
Ip, If, Ia = emd.spectra.frequency_transform(imf, fs, 'nht')






freq_edges, freq_bins = emd.spectra.define_hist_bins(1, 480, 479, 'linear')
hht = emd.spectra.hilberthuang(If, Ia, freq_edges, mode='energy')

plt.figure()
plt.imshow(hht[::-1], aspect='auto')
vizspect(t, freq_edges, hht, 'HHT '+frame, ylims=[1,480])

shht = ndimage.gaussian_filter(hht, 2)
fig = plt.figure(figsize=(10, 6)) 
emd.plotting.plot_hilberthuang(shht, t, freq_bins, fig=fig,freq_lims=(1, 480), log_y=True, cmap='turbo')






