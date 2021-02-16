import os, gc, queue
from struct import unpack
import time
from collections import deque
import numpy as np
import autograd.numpy as nap
import scipy.optimize as op
import scipy.integrate as intg
from autograd import jacobian, hessian
from numpy.linalg import norm, inv
import smbus




class daq:
    def __init__(self):
        self.__name__ = "daq"
        try:
            self.bus = smbus.SMBus(1)
            
            print("bus connected")
        except Exception as e:
            print("ERROR ", e)

        self.devices = []
        self.fs = 3333
        self.dt = 1/self.fs
        self.state = True

        self.odr = 9  #8=1660Hz 9=3330Hz 10=6660Hz
        self.range = [1, 3]     #[16G, 2000DPS]
        for device in range(128):
            try:
                self.bus.read_byte(device)
                if device == 0x6b or device == 0x6a:
                    self.devices.append([device, 0x22, 12, '<hhhhhh', 1])
                self.config(device)
                print("Device Config: ", device)
            except Exception as e:
                #print("ERROR ", e)
                pass

    def config(self, _device):
        if _device == 0x6a or _device == 0x6b:
            _settings = [[0x10, (self.odr<<4 | self.range[0]<<2)],
                         [0x11, (self.odr<<4 | self.range[1]<<2)],
                         [0x12, 0x44]]  #[0x44 is hardcoded acording to LSM6DSO datasheet]

            for _set in _settings:
                try:
                    self.bus.write_byte_data(_device, _set[0], _set[1])
                    
                except Exception as e:
                    print("ERROR: ",e)

    def calibrate(self, _device):
        _sensname = input('Connnect sensor and name it: ')
        qc = queue.Queue()
        print('6 pos calibration')
        for _n in range(6):
            input('Position {}'.format(_n+1))
            _samp1 = self.pull(_device, qc, 5)
        print('3 axis rotation')
        for _n in range(0,6,2):
            input('Rotate 90 deg around axis {}-{}'.format(_n+1,_n+2))
            _samp2 = self.pull(_device, qc, 1)
        _aux = []
        print('Data collection done...') 
        while _q.qsize()>0:
            _d = _q.get()
            _aux.append(unpack('<hhhhhh',bytearray(_d[0:12])))
        _data = np.array(_aux)
        _sensor = {'raw': _data}
        _araw = _data[0:6*_samp1,3:6]
        _graw = _data[:,0:3]

        _acc_m = np.zeros((6, 3))
        for _i in range(6):
            for _j in range(3):
                _acc_m[_i, _j] = np.mean(_accdata[_i*_samp1:(_i+1)*_samp1, _j])
        
        _k = np.zeros((3, 3))
        _b = np.zeros((3))
        _Ti = np.ones((3, 3))

        for _i in range(3):
            _max = _acc_m[:,_i].max(0)
            _min = _acc_m[:,_i].min(0)
            _k[_i, _i] = (_max - _min)/ (2)
            _b[_i] = (_max + _min)/2    
            _Ti[_i, _i-2] = np.arctan(self.acc_m[self.acc_m[:,_i].argmax(0),_i-2] / _max)
            _Ti[_i, _i-1] = np.arctan(self.acc_m[self.acc_m[:,_i].argmax(0),_i-1] / _max)
        _kT = inv(_k.dot(inv(_Ti)))

        _param_a = list(_kT.flatten()) + list(_b.flatten())


        _b = np.mean(_graw[0:6*_samp1,:], axis=0).T
        _gyr_r = _graw[6*_samp1:,:] - _b
        _ang = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                _ang[i, j] = np.abs(intg.trapz(_gyr_r[_samp2*i:_samp2*(i+1), j], dx=self.dt))

        _k = np.zeros((3,3))
        _k[:,0] = _ang[:,_ang[0].argmax()]
        _k[:,1] = _ang[:,_ang[1].argmax()]
        _k[:,2] = _ang[:,_ang[2].argmax()]

        _kT = np.diag([90,90,90])@inv(_k)
        _param_g = list(_kT.flatten()) + list(_b.flatten())


        _res_a = op.minimize(self.funobj_acc, _param_a, method='trust-ncg', jac=jacobian(funobj_acc), hess=hessian(funobj_acc))
        _res_g = op.minimize(self.funobj_gyr, _param_g, method='trust-ncg', jac=jacobian(funobj_gyr), hess=hessian(funobj_gyr))

        _sensor['param_a'] = _res_a.x
        _sensor['param_b'] = _res_b.x

        np.savez('./sensor/{}.npz', _sensor)
        


    def funobj_acc(self, Y):
        _S = np.array(Y[0:9]).reshape((3,3))
        _B = np.array(Y[9:]).reshape((3,1))
        _sum = 0
        for u in _araw:
            _sum += (1 - nap.linalg.norm(_S@(u-_B).T))**2

        return _sum

    def funobj_gyr(self, Y):
        _S = np.array(Y[0:9]).reshape((3,3))
        _B = np.array(Y[9:]).reshape((3,1))
        _sum = 0
        for u in _araw:
            _sum += (_S@(u-_B).T)*self.dt

        return (90 - _sum)**2


    def pull(self, _device, _q, _size):
        i=0
        t0=tf = time.perf_counter()
        while i < _size//self.dt:
            ti=time.perf_counter()
            if ti-tf>=self.dt:
                tf = ti
                i+=1
                _q.put(self.bus.read_i2c_block_data(_device[0],_device[1], _device[2]))
        t1 = time.perf_counter()
        return _size//self.dt

        

    def pulldata(self, _size = 3):
        gc.collect()
        self.q = queue.Queue()
        if _size ==0:
            i=0
            self.state = True
            tf = time.perf_counter()
            while self.state:
                ti=time.perf_counter()
                if ti-tf>=self.dt:
                    tf = ti
                    i+=1
                    self.q.put(self.bus.read_i2c_block_data(self.devices[0][0],self.devices[0][1], self.devices[0][2]) + self.bus.read_i2c_block_data(self.devices[1][0],self.devices[1][1], self.devices[1][2]))
        else:
            i=0
            t0=tf = time.perf_counter()
            while i< _size//self.dt:
                ti=time.perf_counter()
                if ti-tf>=self.dt:
                    tf = ti
                    i+=1
                    self.q.put(self.bus.read_i2c_block_data(self.devices[0][0],self.devices[0][1], self.devices[0][2]) + self.bus.read_i2c_block_data(self.devices[1][0],self.devices[1][1], self.devices[1][2]))
            t1 = time.perf_counter()
            print(t1-t0)
        

        self.savedata(self.q)

    def savedata(self, _q):
        if 'DATA' not in os.listdir():
            os.mkdir('DATA')
        data = []
        while _q.qsize()>0:
            _d = _q.get()
            data.append(unpack('<hhhhhh',bytearray(_d[0:12])) + unpack('<hhhhhh',bytearray(_d[12:24])))
        arr = np.array(data)
        os.chdir('DATA')
        _filename = 'raw_{}.npy'.format(len(os.listdir()))
        np.save(_filename, arr)
        print('{} saved'.format(_filename))
        os.chdir('..')

    


