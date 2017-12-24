# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:09:34 2017
This test file is dependent on vrep.

@author: cz
"""

from scene import Scene
from robot import Robot
import numpy as np

try:
    sc = Scene()
    dynamics = 11
    sc.addRobot(np.float32([[-2, 0, 0], [0, 2/2, 0]]), dynamics)
    sc.addRobot(np.float32([[1, 3, 0], [1.732/2, -1/2, 0]]), dynamics)
    sc.addRobot(np.float32([[0, 0, 0], [-1.732/2, -1/2, 0]]), dynamics)
    
    sc.setADjMatrix(np.uint8([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
    # vrep related
    sc.initVrep()
    objectNames = ['Pioneer_p3dx', 'Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
    sc.setVrepHandles(0, objectNames)
    sc.setVrepHandles(1, objectNames, '#0')
    sc.setVrepHandles(2, objectNames, '#1')
    
    #sc.renderScene(waitTime = 3000)
    tf = 10
    while sc.simulate():
        sc.renderScene(waitTime = int(sc.dt * 1000))
        print("---------------------")
        print("t = %.3f" % sc.t, "s")
        
        sc.plot(0, tf)
        #sc.plot(2, tf)
        #sc.plot(1, tf) 
        if sc.t > tf:
            break
    
        #print('robot 0: ', sc.robots[0].xi.x, ', ', sc.robots[0].xi.y, ', ', sc.robots[0].xi.theta)
        #print('robot 1: ', sc.robots[1].xi.x, ', ', sc.robots[1].xi.y, ', ', sc.robots[1].xi.theta)
        #print('robot 2: ', sc.robots[2].xi.x, ', ', sc.robots[2].xi.y, ', ', sc.robots[2].xi.theta)
        #print('y01: ' + str(sc.robots[1].xi.y - sc.robots[0].xi.y))
        #print('x02: ' + str(sc.robots[2].xi.x - sc.robots[0].xi.x))
    sc.deallocate()
except KeyboardInterrupt:
    sc.deallocate()
except:
    sc.deallocate()
    raise

