import os, gc
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
            bus = smbus.SMBus(1)
        except Exception as e:
            pass

        devices = []
        self.fs = 3330
        self.dt = 1/self.fs

        self.odr = 9  #8=1660Hz 9=3330Hz 10=6660Hz
        self.range = [1, 3]     #[16G, 2000DPS]
        for device in range(128):
            try:
                bus.read_byte(device)
                devices.append(device)
                self.config(device)
                print("Device Config: ", device)
            except: # exception if read_byte fails
                pass

    def config(self, _device):
        if _device == 0x6a or 0x6b:
            _settings = [[0x10, (self.odr<<4 | self.range[0]<<2)],
                         [0x11, (self.odr<<4 | self.range[1]<<2)],
                         [0x12, 0x44]]  #[0x44 is hardcoded acording to LSM6DSO datasheet]

            for _set in _settings:
                try:
                    bus.write_byte_data(_device, _set[0], _set[1])
                except Exception as e:
                    print("ERROR: ",e)
            





