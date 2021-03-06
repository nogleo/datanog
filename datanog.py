import os, gc, queue
from struct import unpack
import time
import numpy as np
import scipy.integrate as intg
from numpy.linalg import norm, inv, pinv
from smbus import SMBus
import sigprocess as sp
import scipy


root = os.getcwd()


class daq:
    def __init__(self, fs=3330):
        self.__name__ = "daq"
        try:
            self.bus = SMBus(1)
            print("bus connected")
        except Exception as e:
            print("ERROR ", e)

        self.dev = []
        self.fs = fs
        self.dt = 1/self.fs
        self.state = 1
        self.raw = 1
        self.G = 9.81
        self.Rot = np.pi
        
        self.odr = 9  #8=1660Hz 9=3330Hz 10=6660Hz
        self.range = [1, 3]     #[16G, 2000DPS]
        for device in range(128):
            try:
                self.bus.read_byte(device)
                if device == 0x6b or device == 0x6a:
                    self.dev.append([device, 0x22, 12, '<hhhhhh', None])
                elif device == 0x36:
                    self.dev.append([device, 0x0C, 2, '>H', None])
                elif device == 0x48:
                    self.dev.append([device, 0x00, 2, '>h', None])
                self.config(device)
                print("Device Config: ", device)
            except Exception as e:
                #print("ERROR ", e)
                pass
        self.N = len(self.dev)

    def config(self, _device):
        _settings = None
        if _device == 0x6a or _device == 0x6b:
            _settings = [[0x10, (self.odr<<4 | self.range[0]<<2 | 1<<1)],
                         [0x11, (self.odr<<4 | self.range[1]<<2)],
                         [0x12, 0x44],
                         [0x13, 1<<1],
                         [0x15, 0b011],
                         [0X17, (0b000 <<5)]]  #[0x44 is hardcoded acording to LSM6DSO datasheet]
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
        os.chdir(root)
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
                if self.dev[_j][-1] != None:
                    _scale = np.load(root+'/sensors/'+self.dev[_j][-1])
                    np.save('rot.npy', sp.fix_outlier(arr*_scale))
                else:
                    np.save('rot*.npy', arr)
            elif str(self.dev[_j][0]) == '106' or str(self.dev[_j][0]) == '107':
                if self.dev[_j][-1] != None:
                    _param = np.load(root+'/sensors/'+self.dev[_j][-1], allow_pickle=True)
                    np.save('gyr{}.npy'.format(self.dev[_j][0]), self.transl(arr[:,0:3], _param['arr_0']))
                    np.save('acc{}.npy'.format(self.dev[_j][0]), self.transl(arr[:,3:6], _param['arr_1']))
                else:
                    np.save('gyr{}*.npy'.format(self.dev[_j][0]), arr[:,0:3])
                    np.save('acc{}*.npy'.format(self.dev[_j][0]), arr[:,3:6])



            elif str(self.dev[_j][0]) == '72':
                peaks,_ = scipy.signal.find_peaks(abs(arr.flatten()),width=2, prominence=5, height=2000, distance=10)
                T = [peaks[0], peaks[peaks>4000][0], peaks[-1]]
                np.save('cur.npy', arr)
                np.save('T.npy', T)


        print('{} saved'.format(_path))

    def to_num(self, _q):
        _data={}
        for _j in range(self.N):
            _data[str(self.dev[_j][0])] = []
        
        while _q.qsize()>0:
            for _j in range(self.N):
                _data[str(self.dev[_j][0])].append(unpack(self.dev[_j][-2], bytearray(_q.get())))
        
        
        return _data
        
    def transl(self, _data, X):
        _T = np.array(X[0:9].reshape((3,3)))
        _b = np.array(X[-3:]).reshape((3,1))
        _data_out = _T@(_data.T-_b)
        
        return _data_out.T

    def calibrate(self, _device):
        os.chdir(root)
        _sensname = input('Connnect sensor and name it: ')
        _sensor = {'name': _sensname}
        self._caldata = []
        print('Iniciando 6 pos calibration')
        self.Ns = int((input('KiloSamples/Position: ') or 6)*1000)

        for _n in range(6):
            input('Position {}'.format(_n+1))
            i=0
            tf = time.perf_counter()
            while i<self.Ns:
                ti=time.perf_counter()
                if ti-tf>=self.dt:
                    tf = ti
                    i+=1
                    self._caldata.append(self.pull(_device))
        self.Nd = int((input('KiloSamples/Rotation: ') or 6)*1000)
        for _n in range(0,6,2):
            input('Rotate 180 deg around axis {}-{}'.format(_n+1,_n+2))
            i=0
            tf = time.perf_counter()
            while i<self.Nd:
                ti=time.perf_counter()
                if ti-tf>=self.dt:
                    tf = ti
                    i+=1
                    self._caldata.append(self.pull(_device))
        
        
        _data = np.array(self._caldata)
        self.acc_raw = _data[0:6*self.Ns,3:6]
        self.gyr_raw = _data[:,0:3]
        np.save('./sensors/'+_sensor['name']+'rawdata.npy', _data)
        print('Calculating calibration parameters. Wait...')
        gc.collect()
        _sensor['acc_p'] = self.calibacc(self.acc_raw)
        gc.collect()
        _sensor['gyr_p'] = self.calibgyr(self.gyr_raw)        
        np.savez('./sensors/'+_sensor['name'], _sensor['gyr_p'], _sensor['acc_p'])
       
        print(_sensor['name']+' saved')
        gc.collect()
        
    
    def calibacc(self, _accdata):        
        #mean values for the 6 positions
        aux=[]
        for ii in range(6):
            aux.append(_accdata[ii*self.Ns:(ii+1)*self.Ns,:].mean(0))
        a_m = np.array(aux).T
        #determination of biases
        aux=[]
        for ii in range(3):
            aux.append((a_m[ii,:].max() + a_m[ii,:].min()) / 2)
        b = np.array(aux, ndmin=2).T

        #unbiased mean values and expected (real) mean values 
        a_mu = a_m-b
        a_mr = np.zeros_like(a_mu)
        for ii in range(3):
            a_mr[ii,a_mu[ii,:].argmax()] = self.G
            a_mr[ii,a_mu[ii,:].argmin()] = -self.G
        #transformation matrix
        T = a_mr@pinv(a_mu)
        
        _param = np.append(T.flatten(), b.T)
       
        return _param
  
    
 
        
    def calibgyr(self, _gyrdata):
        g_s = _gyrdata[0:6*self.Ns,:]            #static gyro data
        g_d = _gyrdata[6*self.Ns:,:]             #dynamic gyro data
        
        b = g_s.mean(0).reshape(3,1)            #bias from mean static measurements
        # integrate dynamic rates to determine total angle
        g_dm = np.zeros((3,3))              

        for ii in range(3):
            g_dm[ii,:] = np.abs(intg.trapz(g_d[ii*self.Nd:(ii+1)*self.Nd,:].T-b,dx=self.dt , axis=1))
        g_dr = np.zeros_like(g_dm)

        for ii in range(3):
            g_dr[ii,g_dm[ii,:].argmax()] = self.Rot
            
        T = g_dr@inv(g_dm)
        
        _param = np.append(T.flatten(), b.T)        
       
        return _param
    
    