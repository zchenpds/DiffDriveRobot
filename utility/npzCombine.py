# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:31:06 2018

@author: cz
"""

import numpy as np

fileName1 = "data1.npz"
fileName2 = "data2.npz"
fileNameNew = "data3"
data1 = np.load(fileName1)
data2 = np.load(fileName2)
dataNew = dict()
for key in data1.keys():
    print(key, ": ", data1[key].shape, ", ", data2[key].shape)
    dataNew[key] =  np.append(data1[key], data2[key], axis = 0)
    print(key, ": ", dataNew[key].shape)
np.savez(fileNameNew, **(dataNew))