# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:14:34 2018

@author: cz
"""
import os
import numpy as np
import sys

dk = 1
dataIn = np.load(os.path.join('data', '2-25', 'data1.npz'))
N = len(dataIn['epi_starts'])
lenObservation = dataIn['observations'].shape[1]

epi_starts = np.array([], dtype = np.bool)
observations = np.zeros((0, lenObservation*2), dtype = np.int8)
actions = np.zeros((0, 2), dtype = np.float32)
actions_1 = np.zeros((0, 2), dtype = np.float32)

k = 0
k1 = None
while k < N:
    if dataIn['epi_starts'][k]:
        if k1 is not None:
            k2 = k
            k3 = k1 + dk
            k4 = k2 - dk
            epi_starts = np.append(epi_starts, True)
            observations = np.append(observations, 
                                     np.append(dataIn['observations'][k1:k4], 
                                               dataIn['observations'][k3:k2], axis = 1), 
                                     axis = 0)
            actions = np.append(actions, dataIn['actions'][k1:k4], axis = 0)
            actions_1 = np.append(actions_1, dataIn['actions'][k3:k2], axis = 0)
        k1 = k
    else:
        epi_starts = np.append(epi_starts, False)
    
    
    sys.stdout.write("\r%.3f%%" % (k/N*100))
    sys.stdout.flush()
    
    k += 1
        
        
np.savez(os.path.join('data', '2-25', 'data1_'), 
         epi_starts = epi_starts,
         observations = observations,
         actions = actions,
         actions_1 = actions_1)