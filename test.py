import queue
import asyncio
import datanog
import time
import numpy as np
import os

dn = datanog.daq()

q = queue.Queue()

def pulldata(_size = 1):
        i=0
        t0=tf = time.perf_counter()
        while i< _size//dn.dt:
            ti=time.perf_counter()
            if ti-tf>=dn.dt:
                tf = ti
                i+=1
                q.put(dn.pull(dn.devices[0]))
        t1 = time.perf_counter()
        print(t1-t0)

def savedata():
    if 'DATA' not in os.listdir():
        os.mkdir('DATA')
    data = []
    while q.qsize()>0:
        data.append(q.get())
    arr = np.array(data)
    os.chdir('DATA')
    np.save('test{}.npy'.format(len(os.listdir())), arr)
    os.chdir('..')

pulldata(2)
savedata()