import queue
import asyncio
import datanog
import time

dn = datanog.daq()

q = queue.Queue(maxsize=10000)

def pulldata(self):
        i=0
        t0=tf = time.perf_counter()
        while i<10000:
            ti=time.perf_counter()
            if ti-tf>=dn.dt:
                tf = ti
                i+=1
                self.q.put(dn.pull(dn.devices[0]))
        t1 = time.perf_counter()
        print(t1-t0)
        data = np.array(q)
        np.save('test.npy', data)