import scipy
import matplotlib.pyplot as plt
import numpy as np

fs = 1660
dt = 1/fs

def fix_outlier(_data):
    _m = _data.mean()
    peaks,_ = scipy.signal.find_peaks(abs(_data.flatten()),width=1, prominence=2, height=3*_m, distance=5))
    for peak in peaks:
        _f = scipy. interpolate.interp1d(np.array([0,9]), np.array([_data[peak-5],_data[peak+5]]).T, kind='linear')
        _data[peak-5:peak+5] = _f(np.arange(0,10)).T
    return _data

def PSD(_data):
    f, Pxx = scipy.signal.welch(_data, fs, nperseg=fs//2, noverlap=fs//4, window='hann', average='median', scaling='spectrum', detrend='linear', axis=0)
    plt.figure()
    plt.subplot(211)
    _t = np.linspace(0, len(_data)*dt, len(_data))
    plt.plot(_t, _data)
    plt.subplot(212)
    plt.plot(f, 20*np.log10(Pxx))
    plt.grid()
    
def spect(_data):
    plt.figure()
    for ii in range(_data.shape[1]):
        plt.subplot(_data.shape[1]*100+10+ii+1)
        f, t, Sxx = scipy.signal.spectrogram(_data[:,ii], fs, axis=0, scaling='spectrum', nperseg=fs//4, noverlap=fs//5, detrend='linear', mode='psd', window='hann')
        Sxx[Sxx==0] = 10**(-20)
        plt.pcolormesh(t, f, 20*np.log10(abs(Sxx)), shading='gouraud', cmap=plt.inferno())#,vmax=20*np.log10(abs(Sxx).max()), vmin=-20*np.log10(abs(Sxx).max()))
        plt.ylim((0, fs//8))
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.tight_layout()
        plt.show()

class imu2body:
       
    def __init__(self, acc, gyr, pos=[0, 0, 0]):
        self.acc_c = self.rate2acc(gyr, pos)
        grv = np.array([[0],[0],[-9.81]])
        q0=ahrs.Quaternion(acc2q(acc[0]))
        imu = ahrs.filters.Complementary(acc=acc, gyr=gyr, frequency=fs, q0=q0, gain=0.001)
        theta = ahrs.QuaternionArray(imu.Q)
        self.th = ahrs.QuaternionArray(imu.Q).to_angles()
        acc_cc = np.zeros_like(acc)
        for ii in range(len(acc)):
            acc_cc[ii,:] = acc[ii,:] + ahrs.Quaternion(imu.Q[ii]).rotate(grv).T 
        self.a = acc_cc
        self.v = FDI(self.a)
        self.d = FDI(self.v)
        self.om = gyr