import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy import signal
import emd
from scipy import ndimage
import ghostipy as gsp
import gc
import ssqueezepy as sq
import os


from sigprocess import *

# import endaq

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

for ii in range(22):
    num = ii
    df = pd.read_csv('DATA/data_{}.csv'.format(num), index_col='t')
    Q = prep_data(df, fs, factor=10, rotroll=50, senslist='BC')

    psd = plt.figure()
    PSD(Q[0], fs, fig=psd)
    PSD(Q[1], fs, fig=psd, linewidth=1)


# %%
df = C
frames = df.columns
S = df.to_numpy()
t = df.index

spect(df, fs)

apply_emd(df, fs)
WSST(df, fs)


c_cwt, _, f_cwt, t_cwt, _ = gsp.cwt(
    S, fs=fs, voices_per_octave=32, method='ola', boundary='periodic', freq_limits=[1, 480])


coefs_cwt, _, f_cwt, t_cwt, _ = gsp.cwt(
    S, timestamps=t, fs=fs, boundary='zeros', method='ola', voices_per_octave=32)
psd_cwt = c_cwt.real**2 + c_cwt.imag**2
psd_cwt /= np.max(psd_cwt)
plt.figure()
ax = plt.yscale('linear')
plt.imshow(psd_cwt, aspect='auto', cmap='turbo',
           extent=[tt[0], tt[-1], ff[0], ff[-1]])
ax.set_xlabel('time')
vizspect(t_cwt, ff, psd_cwt, frame,  fscale='log')
plt.figure()
Pxx = 10*np.log10(np.abs(c_cwt))
plt.pcolormesh(t_cwt, f_cwt, Pxx, **kwargs_dict)

# %%
coefs_wsst, _, f_wsst, t_wsst, _ = gsp.wsst(df.A_Ay.to_numpy(), fs=fs, timestamps=df.index,
                                            freq_limits=[1, 830],
                                            voices_per_octave=32,
                                            boundary='zeros',
                                            method='ola')
psd_wsst = coefs_wsst.real**2 + coefs_wsst.imag**2
vizspect(t_wsst, f_wsst[::-1], psd_wsst, frame, fscale='log')

plt.figure()
plt.imshow(psd_wsst, aspect='auto', cmap='turbo', extent=[
           t_wsst[0], t_wsst[-1], f_wsst[-1], f_wsst[0]])
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


mfreqs = np.array([360, 300, 240, 180, 120, 90, 60, 30, 15, 7.5])
imf, _ = emd.sift.mask_sift(S, mask_freqs=mfreqs/fs,  mask_amp_mode='ratio_sig',
                            ret_mask_freq=True, nphases=8, mask_amp=S.max())
Ip, If, Ia = emd.spectra.frequency_transform(imf, fs, 'nht')


freq_edges, freq_bins = emd.spectra.define_hist_bins(1, 480, 479, 'linear')
hht = emd.spectra.hilberthuang(If, Ia, freq_edges, mode='energy')

plt.figure()
plt.imshow(hht[::-1], aspect='auto')
vizspect(t, freq_edges, hht, 'HHT '+frame, ylims=[1, 480])

shht = ndimage.gaussian_filter(hht, 2)
fig = plt.figure(figsize=(10, 6))
emd.plotting.plot_hilberthuang(
    shht, t, freq_bins, fig=fig, freq_lims=(1, 480), log_y=True, cmap='turbo')


# %%
num = 17
df = pd.read_csv('DATA/data_{}.csv'.format(num), index_col='t')
dp = pd.read_clipboard(index_col=0)
dp.columns = ['acc1_x', 'acc1_y', 'acc1_z', 'acc2_x', 'acc2_y',
              'acc2_z', 'acc3_x', 'acc3_y', 'acc3_z', 'acc4_x', 'acc4_y', 'acc4_z']
dp.index.name = 't'
dp = dp * 9.81
df.B_Az = (df.B_Az + 9.81)*-1

ilocs_df = scipy.signal.argrelextrema(
    abs(df.B_Az.values), np.greater_equal, order=1660*2)[0]
ilocs_dp = scipy.signal.argrelextrema(
    abs(dp.acc.values), np.greater_equal, order=3200*2)[0]

plt.figure()
df.B_Az.plot()
df.iloc[ilocs_df].B_Az.plot(style='.', lw=10, color='red', marker="v")

dp.acc.plot()
dp.iloc[ilocs_dp].acc.plot(style='.', lw=10, color='green', marker="v")


df.iloc[ilocs_df[0]:ilocs_df[-1]].B_Az.plot()
dp.iloc[ilocs_dp[0]:ilocs_dp[-1]].acc.plot()

d1 = df.iloc[ilocs_df[0]:ilocs_df[-1]]
d2 = dp.iloc[ilocs_dp[0]:ilocs_dp[-1]]
d1.index = d1.index - d1.index[0]
d2.index = d2.index - d2.index[0]
d1.B_Az.plot()
d2.acc.plot()

PSD(d1[:6], 1660)
PSD(d2[:6], 3200)
PSD(dp[0:2], 3200)

WSST(d1[:6], 1660)
WSST(d2[:6], 3200)


spect(d1[:6], 1660)
spect(d2[:6], 3200)


ss, tt = scipy.signal.resample(
    df, 10*len(df), t=df.index.to_numpy(), axis=0, window='hann')
FS = 10*fs

B = imu2body(ss, tt, FS)


# %% Teste imu vs 4 acc triax
# num = 7
ii = 0
for num in range(10):
    df = pd.read_csv('teste2211/data_{}.csv'.format(num),
                     index_col='t')[['A_Gx', 'A_Gy', 'A_Gz', 'A_Ax', 'A_Ay', 'A_Az']]
    # df.B_Az = df.B_Az - 9.81
    # cmb = np.array([8.0563e-005,	5.983e-004,	-6.8188e-003])
    # Lb = np.array([5.3302e-018, -7.233e-002, 3.12e-002+2.0e-003])
    # posb = Lb-cmb
    posb = [-8.0563e-005, -7.2928e-002,  4.1019e-002]
    # S=df.to_numpy()
    t = df.index.to_numpy()
    ss, tt = scipy.signal.resample(df, 15*len(df), t=t, axis=0, window='hann')
    FS = 15*fs
    b, a = scipy.signal.cheby1(23, 0.175, 480, fs=FS)
    S = scipy.signal.filtfilt(b, a, ss, axis=0)
    B = imu2body(ss, tt, FS, posb)
    # PSD(B[['Ax', 'Ay', 'Az']], FS)
    # plt.title('Aceleração Linear - IMU {}'.format(num_imu))
    # PSD(B[['thx', 'thy', 'thz']], FS)
    # plt.title('Acelereção Angular - IMU {}'.format(num_imu))

    dp = pd.read_csv('teste2211/tetra_{}.csv'.format(num), header=None, sep='   ', index_col=0,
                     keep_default_na=False, prefix='acc_')[['acc_{}'.format(N) for N in range(3, 70, 6)]]
    dp.index.name = 't'
    dp = dp * 9.81
    # dp.plot()
#     rho =np.array([[-2.8098e-002,	3.3131e-002,	7.0019e-002],
# 	               [ 1.0146e-002,	5.8982e-002,	 8.039e-002],
# 	               [4.7067e-002,   -4.4248e-002,	    4.8519e-002],
# 	               [-3.8802e-002,  -4.3618e-002,	    4.7519e-002]])

    Fs = 3200
    S = np.zeros_like(dp.values)
    b, a = scipy.signal.cheby1(23, 0.175, 480, fs=Fs)
    Sbk = scipy.signal.filtfilt(b, a, dp, axis=0)

    W_dot = B[['alx', 'aly', 'alz']][0.01:].to_numpy()

    rho = np.array([[-4.3211e-002,	 5.2702e-002,   -3.3981e-002],
                    [5.3941e-002,	 4.6393e-002,    4.1519e-002],
                    [4.3049e-002,	-6.4558e-002,   -3.6981e-002],
                    [-4.6081e-002,	-4.6196e-002,	 4.8519e-002]])

    for nacc in range(4):
        r = rho[nacc]
        # nacc=2
        At = np.cross(W_dot, r)
        W = B[['omx', 'omy', 'omz']][0.01:].to_numpy()
        Ac = np.cross(W, np.cross(W, r))
        Ao = B[['Ax', 'Ay', 'Az']][0.01:]
        Ap = pd.DataFrame(Ao + Ac + At, B[0.01:].index)
        Ap['norm'] = np.linalg.norm(Ap, axis=1)
        Abk = pd.DataFrame(dp.to_numpy()[:, 3*nacc:3*(nacc+1)], dp.index)
        Abk['norm'] = np.linalg.norm(Abk, axis=1)

        # Abk.plot()
        psd = plt.figure('exp{}_sens{}'.format(
            num, nacc), dpi=200, figsize=[11, 4])
        PSD(Abk[['norm']], Fs, fig=psd, units='m/s^2', S_ref=1e-6)
        PSD(Ap[['norm']], FS, fig=psd,  units='m/s^2', S_ref=1e-6)
        plt.subplot(211)
        plt.legend(['Piezo', 'IMU'])
        plt.subplot(212)
        plt.ylim(25, 150)
        # plt.xlim((1,500))
        # plt.ylim((-100,20))
        # plt.grid()
        #
        ii += 1
        plt.savefig('./teste2211/teste_comb_{}.pdf'.format(ii))


# plt.close('all')

    # plt.title('Aceleração Linear - ACC {}'.format(num_acc))
    # PSD(state[['p.', 'q.', 'r.']], Fs)
    # plt.title('Aceleração Angular - ACC {}'.format(num_acc))
    # PSD(FDI(FDI(state[['acc_x', 'acc_y', 'acc_z']].to_numpy(), NFFT=Fs//4), NFFT=Fs//4),Fs)
    # PSD(FDI(FDI(state[['p.', 'q.', 'r.']].to_numpy(), NFFT=Fs//4), NFFT=Fs//4),Fs)


# %%
# num = 12
nums = [0, 1, 2, 4, 6, 8, 14, 15, 16, 19]
dur = [[4.5, 7, 2.8, 5.3],
       [3.5, 6, 1.8, 4.3],
       [3, 5.5, 1, 3.5],
       [3, 9, 1.25, 7.25],
       [4.5, 10.5, 2.25, 8.25],
       [3.5, 9.5, 0.625, 6.625],
       [3.5, 12, 0.65, 9.15],
       [4, 11.5, 0.4, 7.9],
       [2, 12, 0.8, 10.8],
       [3.75, 9.5, 2.6, 8.35]]
for ii in range(10):
    num = nums[ii]
    fs1 = 3200
    d1 = pd.read_csv('./301121/teste{}.txt'.format(num), header=None, sep='   ',
                     index_col=0, skiprows=52, keep_default_na=False)[[3, 9, 15, 21, 27, 33]]
    d1.columns = ['acc0_x', 'acc0_y', 'acc0_z', 'acc1_x', 'acc1_y', 'acc1_z']
    d1.index.name = 't'
    #d1 = d1 * 9.81
    # d1.plot()

    d2 = pd.read_csv('./301121/data_{}.csv'.format(num), index_col=0)

    # d2 = pd.DataFrame(ss[:,1:],columns=['acc0_x', 'acc0_y', 'acc0_z', 'acc1_x', 'acc1_y', 'acc1_z'], index=tt)
    # d2.index.name = 't'

    ss, tt = scipy.signal.resample(
        d2.to_numpy(), 10*len(d2), t=d2.index, axis=0, window='hann')
    fs2 = 10*1660
    d2 = imu2body(ss, tt, fs2)

    # d2.alz.plot()
    Ac1 = d1[['acc0_x', 'acc0_y', 'acc1_x', 'acc1_y']]
    Ac1.acc0_x = Ac1.acc0_x*-1
    Ac1['norm0'] = np.linalg.norm(Ac1[['acc0_x', 'acc0_y']], axis=1)
    Ac1['norm1'] = np.linalg.norm(Ac1[['acc1_x', 'acc1_y']], axis=1)
    Ac1 = Ac1[dur[ii][0]:dur[ii][1]]
    Ac1.index = Ac1.index-Ac1.index[0]
    Ac1['norm'] = np.mean((Ac1[['norm0']], Ac1[['norm1']]), axis=0)
    Ac2 = d2[['alz']]*0.05
    Ac2['omz'] = d2[['omz']]**2*0.05
    Ac2['norm'] = np.linalg.norm(Ac2, axis=1)
    Ac2 = Ac2[dur[ii][2]:dur[ii][3]]
    Ac2.index = Ac2.index-Ac2.index[0]
    # Ac2.plot()

    # Ac1.plot()
    # Ac2.plot()
    # plt.close('all')
    psd_comp = plt.figure('teste_gyro_{}'.format(ii), dpi=200, figsize=[11, 4])
    PSD(Ac1[['norm']], fs1, fig=psd_comp,  units='m/s^2', S_ref=1e-6)
    # PSD(Ac1[['norm0']],fs1, fig=psd_comp,  units='m/s^2')
    PSD(Ac2[['norm']], fs2, fig=psd_comp, units='m/s^2', S_ref=1e-6)
    # PSD(Ac1[['norm1']],fs1, fig=psd_comp,  units='m/s^2')
    # plt.ylim([-180,-50])
    plt.subplot(211)
    plt.legend(['Piezo', 'IMU'])

    # plt.savefig('./301121/psd_imuvsbk_{}.png'.format(num), dpi=200)

# %%
for ii in range(1, 13):
    d1 = pd.read_csv('./1602/teste_{}.txt'.format(ii), sep='   ',
                     index_col=0, keep_default_na=False, header=None)[[3]]
    fs1 = 6400
    d1 = d1 * 9.81
    d2 = pd.read_csv('./1602/data_{}.csv'.format(ii), index_col=0)[['B_Az']]
    fs2 = 1660

    peak1, _ = scipy.signal.find_peaks(d1.to_numpy().flatten(), height=(
        12),  width=1, prominence=2,  distance=fs1/2)
    d3 = d1.iloc[peak1[0]:peak1[0]+15*fs1]
    peak2, _ = scipy.signal.find_peaks(d2.to_numpy().flatten(), height=(
        10),  width=1, prominence=2,  distance=fs2/2)
    d4 = d2.iloc[peak2[0]:peak2[0]+15*fs2]
    d3.index = d3.index - d3.index[0]
    d4.index = d4.index - d4.index[0]

    psd_comp = plt.figure('teste_acc_{}'.format(ii), dpi=200, figsize=[11, 4])
    PSD(d3, fs1, fig=psd_comp,  units='m/s^2', S_ref=1e-6)
    PSD(d4*-1-9.81, fs2, fig=psd_comp,  units='m/s^2', S_ref=1e-6)
    plt.subplot(211)
    plt.legend(['Piezo', 'IMU'])

    plt.savefig('./1602/teste_acc_{}.pdf'.format(ii), dpi=200)
