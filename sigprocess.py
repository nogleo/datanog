import scipy
import matplotlib.pyplot as plt
import numpy as np
import ahrs
import pandas as pd
import scipy.signal as signal
import scipy.integrate as intg
from numpy import pi
from scipy.fftpack import fft, ifft, dct, idct, dst, idst, fftshift, fftfreq
from numpy import linspace, zeros, array, pi, sin, cos, exp, arange

fs = 1660
dt = 1/fs
def prep_data(df, fs, fc, factor):
    S = df.to_numpy()   
    S[:,0] = np.unwrap(S[:,0])
    b,a = scipy.signal.cheby1(23, 0.175, fc, fs=fs)
    S[:,2:] = scipy.signal.filtfilt(b, a, S[:,2:], axis=0)
    # S[:,2:] = freqfilt(S[:,2:], fs, fc)
    t = df.index.to_numpy()
    N = len(S)
    ss, tt = scipy.signal.resample(S,factor*N, t=t, axis=0, window='hann')
    ss[:,0] = ss[:,0]%(2*np.pi)
    ss = ss[100:-100,:]
    tt = tt[100:-100]
    fs = factor*fs
    dt = 1/fs
    
    return ss, tt, fs, dt
    

def freqfilt(data, fs, fc):
    data[:,0] = np.unwrap(np.deg2rad(data[:,0]))
    N = len(data)
    ff = fftfreq(N,1/fs)
    k = (abs(ff)<=fc).reshape((N,1))
    Data = fft(data, axis=0)
    Data.real = Data.real*k
    # Data.real = Data.real*k
    data_out = np.real(ifft(Data, axis=0))
    data_out[:,0] = data_out[:,0]%(2*np.pi)
    return data_out
    
    

def fix_outlier(_data):
    _m = _data.mean()
    peaks,_ = scipy.signal.find_peaks(abs(_data.flatten()),width=1, prominence=2, height=3*_m, distance=5)
    for peak in peaks:
        _f = scipy. interpolate.interp1d(np.array([0,9]), np.array([_data[peak-5],_data[peak+5]]).T, kind='linear')
        _data[peak-5:peak+5] = _f(np.arange(0,10)).T
    return _data

# def PSD(_data, fs):
#     f, Pxx = scipy.signal.welch(_data, fs, nperseg=fs//4, noverlap=fs//8, window='hann', average='median', scaling='spectrum', detrend='linear', axis=0)
#     plt.figure()
#     plt.subplot(211)
#     _t = np.linspace(0, len(_data)*dt, len(_data))
#     plt.plot(_t, _data)
#     plt.subplot(212)
#     plt.semilogx(f, 20*np.log10(abs(Pxx)))
#     plt.xlim((1,415))
#     plt.grid()

def PSD(df, fs):
    f, Pxx = scipy.signal.welch(df, fs, nperseg=fs//4, noverlap=fs//8, window='hann', average='mean', scaling='spectrum', detrend=False, axis=0)
    plt.figure()
    plt.subplot(211)
    plt.plot(df)
    plt.legend(df.columns)
    plt.subplot(212)
    plt.semilogx(f, 20*np.log10(abs(Pxx)))
    plt.xlim((1,415))
    plt.grid()   


def FDI(data, factor=1, NFFT=fs//4):
    n = NFFT
    try:
        width = data.shape[1]
    except:
        width = 0
    _data = np.vstack((np.zeros((2*n,width)), data, np.zeros((2*n,width))))
    N = len(_data)
    w = scipy.signal.windows.hann(n).reshape((n,1))
    Data = np.zeros_like(_data, dtype=complex)
    for ii in range(0, N-n, n//2):
        Y = _data[ii:ii+n,:]*w
        k =  (1j*2*np.pi*scipy.fft.fftfreq(len(Y), dt).reshape((n,1)))
        y = (scipy.fft.ifft(np.vstack((np.zeros((factor,width)),scipy.fft.fft(Y, axis=0)[factor:]/(k[factor:]))), axis=0))
        Data[ii:ii+n,:] += y
    return np.real(Data[2*n:-2*n,:])
    
# def spect(df,fs, dbmin=80):
          
#     plt.figure()
#     if len(_data.shape)<2:
#         _data = _data.reshape((len(_data),1))
#     kk = _data.shape[1]           
#     for ii in range(kk):
#         plt.subplot(kk*100+10+ii+1)
#         f, t, Sxx = scipy.signal.spectrogram(_data[:,ii], fs=fs, axis=0, scaling='spectrum', nperseg=fs//4, noverlap=fs//8, detrend='linear', mode='psd', window='hann')
#         Sxx[Sxx==0] = 10**(-20)
#         plt.pcolormesh(t, f, 20*np.log10(abs(Sxx)), shading='auto', cmap=plt.inferno(),vmax=20*np.log10(abs(Sxx)).max(), vmin=20*np.log10(abs(Sxx)).max()-dbmin)
#         plt.ylim((0, 300))
#         plt.colorbar()
#         plt.ylabel('Frequency [Hz]')
#         plt.xlabel('Time [sec]')
#         plt.tight_layout()
#         plt.show()

def spect(df,fs, dbmin=80):
    for frame in df:
        plt.figure()
        f, t, Sxx = scipy.signal.spectrogram(df[frame], fs=fs, axis=0, scaling='spectrum', nperseg=fs//4, noverlap=fs//8, detrend=False, mode='psd', window='hann')
        Sxx[Sxx==0] = 10**(-20)
        plt.pcolormesh(t, f, 20*np.log10(abs(Sxx)), shading='gouraud',  cmap=plt.cm.Spectral_r,vmax=20*np.log10(abs(Sxx)).max(), vmin=20*np.log10(abs(Sxx)).max()-dbmin)
        plt.ylim((0, 415))
        plt.colorbar()
        plt.title(frame)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.tight_layout()
        plt.show()



    
    


        
def FDD(_data, factor=1, NFFT=fs):
    N = len(_data)
    try:
        width = _data.shape[1]
    except:
        _data = _data.reshape((N,1))
        width = 1
    n = NFFT
    w = signal.windows.hann(n).reshape((n,1))
    Data = np.zeros_like(_data, dtype=complex)
    for ii in range(0, N-n, n//2):
        Y = _data[ii:ii+n,:]*w
        k =  (1j*2*pi*fftfreq(len(Y), dt).reshape((n,1)))
        y = (ifft(np.vstack((np.zeros((factor,width)),fft(Y, axis=0)[factor:]*(k[factor:]))), axis=0))
        Data[ii:ii+n,:] += y
    return np.real(Data)        

def TDI(_data): 
    N = len(_data)
    if len(_data.shape)==1:
        _data = _data.reshape((N,1))
    _data = zmean(_data)
    _dataout = np.zeros_like(_data)
    _dataout[0,:] = _data[0,:]*dt/2
    for ii in range(1,N):
        _dataout[ii,:] = intg.simpson(_data[0:ii,:], dx=dt, axis=0)
    return _dataout

def zmean(_data):
    return np.real(ifft(np.vstack((np.zeros((2,_data.shape[1])),fft(_data, axis=0)[2:])), axis=0))
       
def imu2body(df, t, fs, pos=[0, 0, 0]):
    gyr = df[:,0:3]
    acc = df[:,3:]
    grv = np.array([[0],[0],[-9.81]])
    alpha = FDD(gyr)
    accc = acc + np.cross(gyr,np.cross(gyr,pos)) + np.cross(alpha,pos)
    q0=ahrs.Quaternion(ahrs.common.orientation.acc2q(accc[0]))
    imu = ahrs.filters.Complementary(acc=accc, gyr=gyr, frequency=fs, q0=q0, gain=0.0001)
    theta = ahrs.QuaternionArray(imu.Q).to_angles()
    
    acccc = np.zeros_like(accc)
    for ii in range(len(acc)):
        acccc[ii,:] = accc[ii,:] + ahrs.Quaternion(imu.Q[ii]).rotate(grv).T
    
    
    v = FDI(acccc)
    d = FDI(v)
    ah = {}
    ah['Dx'] = d[:,0]
    ah['Dy'] = d[:,1]
    ah['Dz'] = d[:,2]
    ah['Vx'] = v[:,0]
    ah['Vy'] = v[:,1]
    ah['Vz'] = v[:,2]
    ah['Ax'] = acccc[:,0]
    ah['Ay'] = acccc[:,1]
    ah['Az'] = acccc[:,2]
    ah['thx'] = theta[:,0]
    ah['thy'] = theta[:,1]
    ah['thz'] = theta[:,2]
    ah['omx'] = gyr[:,0]
    ah['omy'] = gyr[:,1]
    ah['omz'] = gyr[:,2]
    ah['alx'] = alpha[:,0]
    ah['aly'] = alpha[:,1]
    ah['alz'] = alpha[:,2]
    
    
    

    dataFrame = pd.DataFrame(ah, t)
    return dataFrame