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





class daq:
    def __init__(self):
        self.__name__ = "daq"
        try:
            self.bus = SMBus(1)
            print("bus connected")
        except Exception as e:
            print("ERROR ", e)

        self.dev = []
        self.fs = 1660
        self.dt = 1/self.fs
        self.state = 1
        self.raw = 1
        self.G = 1
        self.root = os.getcwd()
        self.odr = 8  #8=1660Hz 9=3330Hz 10=6660Hz
        self.range = [1, 3]     #[16G, 2000DPS]
        for device in range(128):
            try:
                self.bus.read_byte(device)
                if device == 0x6b or device == 0x6a:
                    self.dev.append([device, 0x22, 12, '<hhhhhh'])
                elif device == 0x36:
                    self.dev.append([device, 0x0C, 2, '>H'])
                elif device == 0x48:
                    self.dev.append([device, 0x00, 2, '>h'])
                self.config(device)
                print("Device Config: ", device)
            except Exception as e:
                #print("ERROR ", e)
                pass
        self.N = len(self.dev)

    def config(self, _device):
        _settings = None
        if _device == 0x6a or _device == 0x6b:
            _settings = [[0x10, (self.odr<<4 | self.range[0]<<2)],
                         [0x11, (self.odr<<4 | self.range[1]<<2)],
                         [0x12, 0x44]]  #[0x44 is hardcoded acording to LSM6DSO datasheet]
            for _set in _settings:
                try:
                    self.bus.write_byte_data(_device, _set[0], _set[1])
                    
                except Exception as e:
                    print("ERROR: ",e)
        elif _device == 0x48:
            _config = (3<<9 | 0<<8 | 4<<5 | 3)
            _settings = [0x01, [_config>>8 & 0xFF, _config & 0xFF]]
            try:
                self.bus.write_i2c_block_data(_device, _settings[0], _settings[1])
            except Exception as e:
                    print("ERROR: ",e)
        

    def pull(self, _device):
       return unpack(_device[3], bytearray(self.bus.read_i2c_block_data(_device[0],_device[1], _device[2])))

    def pulldata(self, _size = 1):
        self.q = queue.Queue()
        gc.collect()
        self.state = 1
        if int(_size) == 0:
                
            t0=tf = time.perf_counter()
            while self.state:
                ti=time.perf_counter()
                if ti-tf>=self.dt:
                    tf = ti                    
                    for _j in range(self.N):
                        try:
                            self.q.put(self.bus.read_i2c_block_data(self.dev[_j][0],self.dev[_j][1],self.dev[_j][2]))
                        except Exception as e:
                            self.q.put((0,)*self.dev[_j][2])
                            print(e)    
                t1 = time.perf_counter()
            print(t1-t0)
        else:
                Ns = int(_size)//self.dt
                i=0
                t0=tf = time.perf_counter()
                while i < Ns:
                    ti=time.perf_counter()
                    if ti-tf>=self.dt:
                        tf = ti
                        i+=1
                        
                        for _j in range(self.N):
                            try:
                                self.q.put(self.bus.read_i2c_block_data(self.dev[_j][0],self.dev[_j][1],self.dev[_j][2]))
                            except Exception as e:
                                self.q.put((0,)*self.dev[_j][2])
                                print(e)
                    t1 = time.perf_counter()
                print(t1-t0)
        return self.q  

    def savedata(self, _q):
        os.chdir(self.root)
        if 'DATA' not in os.listdir():
            os.mkdir('DATA')
        os.chdir('DATA')
        _path = 'data_{}'.format(len(os.listdir()))
        os.mkdir(_path)
        os.chdir(_path)
        
        data = self.to_num(_q)



        for _j in range(self.N):
            arr = np.array(data[str(self.dev[_j][0])])    
            if str(self.dev[_j][0]) == '54':
                np.save('rot.npy', arr)
            elif str(self.dev[_j][0]) == '106' or str(self.dev[_j][0]) == '107':
                np.save('gyr{}.npy'.format(str(_j)), arr[:,0:3])
                np.save('acc{}.npy'.format(str(_j)), arr[:,3:6])
            elif str(self.dev[_j][0]) == '72':
                np.save('cur.npy', arr)


        print('{} saved'.format(_path))

    def to_num(self, _q):
        _data={}
        for _j in range(self.N):
            _data[str(self.dev[_j][0])] = []
        
        while _q.qsize()>0:
            for _j in range(self.N):
                _data[str(self.dev[_j][0])].append(unpack(self.dev[_j][-1], bytearray(_q.get())))
        
        return _data
        


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
        
        
        _data = np.array(self._caldata)
        self.acc_raw = _data[0:6*self._nsamp,3:6]
        self.gyr_raw = _data[:,0:3]
        #np.save('./sensors/'+_sensor['name']+'rawdata.npy', _data)
        #print(_sensor['name']+'rawdata saved')
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
        _param = np.append(_kT.flatten(), _b.T)
        _jac = jacobian(self.accObj)
        _hes = hessian(self.accObj)
        _res = op.minimize(self.accObj, _param, method='trust-ncg', jac=_jac, hess=_hes)
        return _res.x
  
    
    def accObj(self, X):
        _NS = nap.array(X[0:9].reshape((3,3)))
        _b = nap.array(X[-3:])
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
        _param = np.append(_kT.flatten(), _b.T)        
        _jac = jacobian(self.gyrObj)
        _hes = hessian(self.gyrObj)
        _res = op.minimize(self.gyrObj, _param, method='trust-ncg', jac=_jac, hess=_hes)
        return _res.x
    
    def gyrObj(self,Y):
        _NS = nap.array(Y[0:9].reshape((3,3)))
        _b = nap.array(Y[-3:])
        sum = 0
        for u in self.rates:
            sum += _NS@(u-_b).T*self.dt
       
    
        return (90 - nap.abs(sum)).sum()**2

