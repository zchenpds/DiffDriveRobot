# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:41:44 2018

@author: cz
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

def saturate(dxp, dyp, dxypMax):
    dxyp = (dxp**2 + dyp**2)**0.5
    dxp = dxp * np.where(dxyp > dxypMax, 1 / dxyp * dxypMax, 1)
    dyp = dyp * np.where(dxyp > dxypMax, 1 / dxyp * dxypMax, 1)
    return dxp, dyp

K3 = -0.15
step = 0.2
pijd0 = 1

X, Y = np.meshgrid(np.arange(0, 5.2, step), np.arange(0, 4, step))
# others = [[1, 1], [1, 2], [2.732, 1], [2.732, 2]]
# others = [[1, 1], [1+1.732/2, 1.5], [2.732, 1]]
others = [[2, 2], [3, 2]]
U = 0
V = 0
for robot in others:
    pijx = X - robot[0]
    pijy = Y - robot[1]
    pij0 = (pijx**2 + pijy**2)**0.5 # the norm of pij
    pij0 = np.where(pij0 < 1e-4, 1, pij0)
    
    
    tauij0 = 2 * (pij0**4 - pijd0**4) / pij0**3
    
    U += tauij0 * pijx / pij0
    V += tauij0 * pijy / pij0

U, V = saturate(U * K3, V * K3, 0.7)
plt.figure()
plt.title('Vector field')
Q = plt.quiver(X, Y, U, V, units='width')
qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.axes().set_aspect('equal', 'datalim')