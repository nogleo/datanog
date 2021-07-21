#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 18:16:25 2021

@author: nog
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import signal

from sigprocess import *
num = 6
df = pd.read_csv('DATA/data_{}.csv'.format(num), index_col='t')

    
PSD(df.to_numpy()[:,5:])

psd = get_psd(df,0.50)
psd.plot()


df[df.columns[-6:]].plot()
df.index
