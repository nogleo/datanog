import queue
import asyncio
import datanog
import time
import numpy as np
import os

dn = datanog.daq()


dn.pulldata(1)

dn.pulldata2(1)
