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
PSD(df[['B_Ax', 'B_Ay', 'B_Az']], fs)
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




# %%
num=17
df = pd.read_csv('DATA/data_{}.csv'.format(num), index_col='t')[['B_Az']]
dp = pd.read_clipboard(index_col=0)
dp.columns = ['acc1_x', 'acc1_y', 'acc1_z', 'acc2_x', 'acc2_y', 'acc2_z', 'acc3_x', 'acc3_y', 'acc3_z', 'acc4_x', 'acc4_y', 'acc4_z']
dp.index.name = 't'
dp = dp * 9.81
df.B_Az = (df.B_Az + 9.81 )*-1

ilocs_df = scipy.signal.argrelextrema(abs(df.B_Az.values), np.greater_equal, order=1660*2)[0]
ilocs_dp = scipy.signal.argrelextrema(abs(dp.acc.values), np.greater_equal, order=3200*2)[0]

plt.figure()
df.B_Az.plot()
df.iloc[ilocs_df].B_Az.plot(style='.', lw=10, color='red', marker="v");

dp.acc.plot()
dp.iloc[ilocs_dp].acc.plot(style='.', lw=10, color='green', marker="v");


df.iloc[ilocs_df[0]:ilocs_df[-1]].B_Az.plot()
dp.iloc[ilocs_dp[0]:ilocs_dp[-1]].acc.plot()

d1 = df.iloc[ilocs_df[0]:ilocs_df[-1]]
d2 = dp.iloc[ilocs_dp[0]:ilocs_dp[-1]]
d1.index = d1.index - d1.index[0]
d2.index = d2.index - d2.index[0]
d1.B_Az.plot()
d2.acc.plot()

PSD(d1[:6],1660)
PSD(d2[:6],3200)
PSD(dp[0:2],3200)

WSST(d1[:6], 1660)
WSST(d2[:6], 3200)


spect(d1[:6], 1660)
spect(d2[:6], 3200)


ss, tt = scipy.signal.resample(df,10*len(df), t=df.index.to_numpy(), axis=0, window='hann')
FS = 10*fs

B = imu2body(ss,tt,FS)



# %% Teste imu vs 4 acc triax
# num_imu = 9
for num_imu in range(10):
    df = pd.read_csv('teste2211/data_{}.csv'.format(num_imu), index_col='t')[['A_Gx','A_Gy', 'A_Gz', 'A_Ax', 'A_Ay', 'A_Az']]
    # df.B_Az = df.B_Az - 9.81
    # cmb = np.array([8.0563e-005,	5.983e-004,	-6.8188e-003])
    # Lb = np.array([5.3302e-018, -7.233e-002, 3.12e-002+2.0e-003])
    # posb = Lb-cmb
    posb = [-8.0563e-005,	-7.2928e-002,	4.0019e-002,]
    # b,a = scipy.signal.cheby1(23, 0.175, 480, fs=fs)
    # S = scipy.signal.filtfilt(b, a, df, axis=0)
    S=df.to_numpy()
    t = df.index.to_numpy()
    ss, tt = scipy.signal.resample(S,10*len(S), t=t, axis=0, window='hann')
    FS = 10*fs
    B = imu2body(ss, tt, FS, posb)
    PSD(B[['Ax', 'Ay', 'Az']], FS)
    PSD(B[['alx', 'aly', 'alz']], FS)
    plt.title('IMU {}'.format(num_imu))

# num_acc = 9
for num_acc in range(10):
    dp = pd.read_csv('teste2211/tetra_{}.csv'.format(num_acc),header=None,sep='   ', index_col=0, keep_default_na=False, prefix= 'acc_')[['acc_{}'.format(N) for N in range(3,70,6)]]
    dp.index.name = 't'
    dp = dp * 9.81
    # dp.plot()
#     rho =np.array([[-2.8098e-002,	3.3131e-002,	7.0019e-002],
# 	               [ 1.0146e-002,	5.8982e-002,	 8.039e-002],
# 	               [4.7067e-002,   -4.4248e-002,	    4.8519e-002],
# 	               [-3.8802e-002,  -4.3618e-002,	    4.7519e-002]])
    
    rho = np.array([[ 5.0093e-002,	 4.3560e-002,	 3.5519e-002],
 	                [ 4.1802e-002,	-5.4928e-002,	-3.3981e-002],
 	                [-4.2797e-002,	-4.4248e-002,	 4.1519e-002],
 	                [-4.1703e-002,	 4.3072e-002,	-3.9981e-002]])*20
	      
	
    C = np.zeros((12,12))
    for ii in range(4):
        gg = [[         0,  rho[ii,2], -rho[ii,1],           0, -rho[ii,0],  rho[ii,1],  rho[ii,2],  rho[ii,2],          0],
              [-rho[ii,2],          0,  rho[ii,0],  -rho[ii,1],          0,  rho[ii,0],          0,          0,  rho[ii,2]],
              [ rho[ii,1], -rho[ii,0],          0,  -rho[ii,2], -rho[ii,2],          0,  rho[ii,0],  rho[ii,0],  rho[ii,1]]]
        G = np.hstack((np.identity(3), np.array(gg)))
        C[ii*3:(ii*3)+3,:] = G
    Cinv = np.linalg.inv(C)
    Fs = 3200
    S = np.zeros_like(dp.values)
    b,a = scipy.signal.cheby1(23, 0.175, 300, fs=Fs)
    Sbk = scipy.signal.filtfilt(b, a, dp, axis=0)
    # Sbk = dp.values
    for jj in range(len(dp)):
        S[jj] = (Cinv@Sbk[jj].reshape((12,1))).T
    state = pd.DataFrame(S, index=dp.index, columns=['acc_x', 'acc_y', 'acc_z', 'p.', 'q.', 'r.', 'p²', 'q²', 'r²', 'pq', 'pr', 'qr'])
    
    PSD(state[['acc_x', 'acc_y', 'acc_z']], 3200)
    PSD(state[['p.', 'q.', 'r.']], 3200)
    plt.title('ACC {}'.format(num_acc))


