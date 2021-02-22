import os, gc, queue
from struct import unpack
import time
import numpy as np
import autograd.numpy as nap
import scipy.optimize as op
import scipy.integrate as intg
from autograd import jacobian, hessian
from numpy.linalg import norm, inv
from smbus import SMBus
dev = []
class daq:
    def __init__(self):
        self.__name__ = "daq"
        try:
            self.bus = SMBus(1)
            print("bus connected")
        except Exception as e:
            print("ERROR ", e)

        self.devices = []
        self.fs = 3330
        self.dt = 1/self.fs
        self.state = True
        self.G = 1

        self.odr = 9  #8=1660Hz 9=3330Hz 10=6660Hz
        self.range = [1, 3]     #[16G, 2000DPS]
        for device in range(128):
            try:
                self.bus.read_byte(device)
                if device == 0x6b or device == 0x6a:
                    dev.append([device, 0x22, 12, '<hhhhhh'])
                if device == 0x36:
                    dev.append([device, 0x0C, 2, '>H'])
                self.config(device)
                print("Device Config: ", device)
            except Exception as e:
                #print("ERROR ", e)
                pass
        self.N = len(dev)

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
        _sensor = {'name': _sensname}
        self._caldata = []
        print('Iniciando 6 pos calibration')
        self._nsamp = int(input('Number of Samples/Position: ') or 3/self.dt)

        for _n in range(6):
            input('Position {}'.format(_n+1))
            i=0
            tf = time.perf_counter()
            while i<self._nsamp:
                ti=time.perf_counter()
                if ti-tf>=self.dt:
                    tf = ti
                    i+=1
                    self._caldata.append(self.pull(_device))
        self._gsamps = int(input('Number of Samples/Rotation: ') or 1/self.dt)
        for _n in range(0,6,2):
            input('Rotate 90 deg around axis {}-{}'.format(_n+1,_n+2))
            i=0
            tf = time.perf_counter()
            while i<self._gsamps:
                ti=time.perf_counter()
                if ti-tf>=self.dt:
                    tf = ti
                    i+=1
                    self._caldata.append(self.pull(_device))
        
        self._aux = []
        print('Data collection done...')
        for _d in self._caldata:
            self._aux.append(unpack('<hhhhhh',bytearray(_d)))
        _data = np.array(self._aux)
        self.acc_raw = _data[0:6*self._nsamp,3:6]
        self.gyr_raw = _data[:,0:3]
        np.save('./sensors/'+_sensor['name']+'rawdata.npy', _data)
        print(_sensor['name']+'rawdata saved')
        print('Calculating calibration parameters. Wait...')
        gc.collect()
        _sensor['acc_p'] = self.calibacc(self.acc_raw)
        gc.collect()
        _sensor['gyr_p'] = self.calibgyr(self.gyr_raw)        
        np.savez('./sensors/'+_sensor['name'], _sensor['gyr_p'], _sensor['acc_p'])
       
        os.chdir('..')
        gc.collect()
        return _sensor
    
    def calibacc(self, _accdata):
        _k = np.zeros((3, 3))
        _b = np.zeros((3))
        _Ti = np.ones((3, 3))
        
        self.acc_m = np.zeros((6, 3))
        for _i in range(6):
            for _j in range(3):
                self.acc_m[_i, _j] = np.mean(_accdata[_i*self._nsamp:(_i+1)*self._nsamp, _j])

        
        for _i in range(3):
            _max = self.acc_m[:,_i].max(0)
            _min = self.acc_m[:,_i].min(0)
            _k[_i, _i] = (_max - _min)/ (2*self.G)
            _b[_i] = (_max + _min)/2    
            _Ti[_i, _i-2] = np.arctan(self.acc_m[self.acc_m[:,_i].argmax(0),_i-2] / _max)
            _Ti[_i, _i-1] = np.arctan(self.acc_m[self.acc_m[:,_i].argmax(0),_i-1] / _max)
        _kT = inv(_k.dot(inv(_Ti)))
        _param = np.append(np.append(np.append(_kT.diagonal(), _b.T), _kT[np.tril(_kT, -1) != 0]), _kT[np.triu(_kT, 1) != 0])
        _jac = jacobian(self.accObj)
        _hes = hessian(self.accObj)
        _res = op.minimize(self.accObj, _param, method='trust-ncg', jac=_jac, hess=_hes)
        return _res.x
  
    
    def accObj(self, X):
        _NS = nap.array([[X[0], X[6], X[7]], [X[8], X[1], X[9]], [X[10], X[11], X[2]]])
        _b = nap.array([X[3], X[4], X[5]])
        _sum = 0
        for u in self.acc_m:
            _sum += (self.G - nap.linalg.norm(_NS@(u-_b).T))**2

        return _sum
        
    def calibgyr(self, _gyrdata):
        _gyr_s = _gyrdata[0:6*self._nsamp,:]
        _b = np.mean(_gyr_s, axis=0).T
        _gyr_d = _gyrdata[6*self._nsamp:,:] 
        _gyr_r = _gyr_d - _b
        _ang = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                _ang[i, j] = np.abs(intg.trapz(_gyr_r[self._gsamps*i:self._gsamps*(i+1), j], dx=self.dt))

        _n = _ang.argmax(axis=0)

        self.rates = np.zeros((self._gsamps,3))
        for i in range(3):
            self.rates[:,i] = _gyr_d[self._gsamps*_n[i]:self._gsamps*(_n[i]+1), i]

        _k = np.zeros((3,3))
        _k[:,0] = _ang[:,_ang[0].argmax()]
        _k[:,1] = _ang[:,_ang[1].argmax()]
        _k[:,2] = _ang[:,_ang[2].argmax()]

        _kT = np.diag([90,90,90])@inv(_k)
        
        _param = np.append(np.append(np.append(_kT.diagonal(), _b.T), _kT[np.tril(_kT, -1) != 0]), _kT[np.triu(_kT, 1) != 0])
        _jac = jacobian(self.gyrObj)
        _hes = hessian(self.gyrObj)
        _res = op.minimize(self.gyrObj, _param, method='trust-ncg', jac=_jac, hess=_hes)
        return _res.x
    
    def gyrObj(self,Y):
        _NS = nap.array([[Y[0], Y[6], Y[7]], [Y[8], Y[1], Y[9]], [Y[10], Y[11], Y[2]]])
        _b = nap.array([Y[3], Y[4], Y[5]])
        sum = 0
        for u in self.rates:
            sum += _NS@(u-_b).T*self.dt
       
    
        return (90 - nap.abs(sum)).sum()**2


    def pull(self, _device):
       return self.bus.read_i2c_block_data(_device[0],_device[1], _device[2])

        

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

    def pulldata2(self, _size = 3):
        dev = self.devices
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
                    for _j in range(self.N):
                        self.q.put(self.bus.read_i2c_block_data(self.devices[_j][0],self.devices[_j][1], self.devices[_j][2]))
        else:
            i=0
            t0=tf = time.perf_counter()
            while i< _size//self.dt:
                ti=time.perf_counter()
                if ti-tf>=self.dt:
                    tf = ti
                    i+=1
                    _aux = []
                    for _j in range(self.N):
                       _aux += self.bus.read_i2c_block_data(dev[_j][0],dev[_j][2],dev[_j][2])
                    self.q.put(_aux)
            t1 = time.perf_counter()
            print(t1-t0)

        

        

    def savedata(self, _q):
        if 'DATA' not in os.listdir():
            os.mkdir('DATA')
        data = []
        dev = self.devices
        while _q.qsize()>0:
            _d = _q.get()
            _aux = []
            i=0
            for j in range(self.N):
                _aux+=unpack(dev[j][3], bytearray(_d[i:i+int(dev[j][2])]))
                i += int(dev[j][2]) 
            data.append(_aux)
        arr = np.array(data)
        os.chdir('DATA')
        _filename = 'raw_{}.npy'.format(len(os.listdir()))
        np.save(_filename, arr)
        print('{} saved'.format(_filename))
        os.chdir('..')

    def savedata2(self, _q):
        if 'DATA' not in os.listdir():
            os.mkdir('DATA')
        data={}
        for _j in range(self.N):
            data[str(self.devices[_j][0])] = []
        
        while _q.qsize()>0:
            for _j in range(self.N):
                data[str(self.devices[_j][0])].append(unpack(self.devices[_j][-1], bytearray(_q.get())))
            
        arr = np.array(data)
        os.chdir('DATA')
        _filename = 'raw_{}.npy'.format(len(os.listdir()))
        np.save(_filename, arr)
        print('{} saved'.format(_filename))
        os.chdir('..')

    


