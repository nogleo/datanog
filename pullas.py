import datanog
from time import sleep
dn = datanog.daq()


while True:
    sleep(0.5)
    print(dn.pull(datanog.dev[0]))