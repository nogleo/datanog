import os, gc, queue
from struct import unpack
import time
from collections import deque
import numpy as np
import autograd.numpy as nap
import scipy.optimize as op
import scipy.integrate as intg
import autograd
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
        self.fs = 3330
        self.dt = 1/self.fs

        self.odr = 9  #8=1660Hz 9=3330Hz 10=6660Hz
        self.range = [1, 3]     #[16G, 2000DPS]
        for device in range(128):
            try:
                self.bus.read_byte(device)
                if device == 0x6b or device == 0x6a:
                    self.devices.append([device, 0x22, 12, '<hhhhhh'])
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

    def pull(self):
        return self.bus.read_i2c_block_data(self.devices[0][0],self.devices[0][1], self.devices[0][2])) + self.bus.read_i2c_block_data(self.devices[1][0],self.devices[1][1], self.devices[1][2]))
        
    def pulldata(self, _size = 3):
        self.q = queue.Queue()
        i=0
        t0=tf = time.perf_counter()
        while i< _size//self.dt:
            ti=time.perf_counter()
            if ti-tf>=self.dt:
                tf = ti
                i+=1
                self.q.put(self.pull())
        t1 = time.perf_counter()
        print(t1-t0)
        self.savedata(self.q)

    def savedata(self, _q):
        if 'DATA' not in os.listdir():
            os.mkdir('DATA')
        data = []
        while _q.qsize()>0:
            data.append(_q.get())
        arr = np.array(data)
        os.chdir('DATA')
        np.save('test{}.npy'.format(len(os.listdir())), arr)
        print('file saved')
        os.chdir('..')

