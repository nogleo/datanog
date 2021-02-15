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
import asyncio



class daq:
    def __init__(self):
        self.__name__ = "daq"
        try:
            self.bus1 = smbus.SMBus(1)
            self.bus2 = smbus.SMBus(6)
            print("buses connected")
        except Exception as e:
            print("ERROR ", e)

        self.devices = []
        self.fs = 3333
        self.dt = 1/self.fs

        self.odr = 9  #8=1660Hz 9=3330Hz 10=6660Hz
        self.range = [1, 3]     #[16G, 2000DPS]
        for device in range(128):
            try:
                self.bus1.read_byte(device)
                if device == 0x6b or device == 0x6a:
                    self.devices.append([device, 0x22, 12, '<hhhhhh', 1])
                self.config(device)
                print("Device Config: ", device)
            except Exception as e:
                #print("ERROR ", e)
                pass

            try:
                self.bus2.read_byte(device)
                if device == 0x6b or device == 0x6a:
                    self.devices.append([device, 0x22, 12, '<hhhhhh', 2])
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
                    if _device[4] == 1:
                        self.bus1.write_byte_data(_device, _set[0], _set[1])
                    elif _device[4] == 2:
                        self.bus2.write_byte_data(_device, _set[0], _set[1])

                except Exception as e:
                    print("ERROR: ",e)

    async def pulldata1(self, _size = 3):
        self.q1 = queue.Queue()
        i=0
        t0=tf = time.perf_counter()
        while i < _size//self.dt:
            ti=time.perf_counter()
            if ti-tf>=self.dt:
                tf = ti
                i+=1
                self.q1.put(self.bus1.read_i2c_block_data(self.devices[0][0],self.devices[0][1], self.devices[0][2]))
        t1 = time.perf_counter()
        print(t1-t0)
        
        if 'DATA' not in os.listdir():
            os.mkdir('DATA')
        data = []
        while self.q1.qsize()>0:
            _d1 = self.q1.get()
            data.append(unpack(self.devices[0][3],bytearray(_d1[0:12])))
        arr = np.array(data)
        os.chdir('DATA')
        np.save('bus1_{}.npy'.format(len(os.listdir())), arr)
        print('file saved')
        os.chdir('..')

    async def pulldata2(self, _size = 3):
        self.q2 = queue.Queue()
        i=0
        t0=tf = time.perf_counter()
        while i < _size//self.dt:
            ti=time.perf_counter()
            if ti-tf>=self.dt:
                tf = ti
                i+=1
                self.q2.put(self.bus2.read_i2c_block_data(self.devices[1][0],self.devices[1][1], self.devices[1][2]))
        t1 = time.perf_counter()
        print(t1-t0)
        
        if 'DATA' not in os.listdir():
            os.mkdir('DATA')
        data = []
        while self.q2.qsize()>0:
            _d2 = self.q2.get()
            data.append(unpack(self.devices[0][3],bytearray(_d2[0:12])))
        arr = np.array(data)
        os.chdir('DATA')
        np.save('bus2_{}.npy'.format(len(os.listdir())), arr)
        print('file saved')
        os.chdir('..')


    async def dualcollect(self):
        await asyncio.gather(
            self.pulldata1(),
            self.pulldata2()
        )
        

    async def runn(self):
        asyncio.run(self.dualcollect())


    '''
    def pull(self):
        return self.bus.read_i2c_block_data(self.devices[0][0],self.devices[0][1], self.devices[0][2]) + self.bus.read_i2c_block_data(self.devices[1][0],self.devices[1][1], self.devices[1][2])
        
    def pulldata(self, _size = 3):
        gc.collect()
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
            _d = _q.get()
            data.append(unpack(self.devices[0][3],bytearray(_d[0:12])) + unpack(self.devices[0][3],bytearray(_d[12:24])))
        arr = np.array(data)
        os.chdir('DATA')
        np.save('test{}.npy'.format(len(os.listdir())), arr)
        print('file saved')
        os.chdir('..')'''

    


