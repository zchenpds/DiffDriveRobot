# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:09:34 2017
This test file is dependent on vrep.

@author: cz
"""

from scene import Scene
from sceneplot import ScenePlot
# from robot import Robot
import numpy as np
# from data import Data

def generateData(**kwargs):
    sc = Scene(recordData = True)
    sp = ScenePlot(sc)
    try:
        #dynamics = 20
        arg2 = np.float32([.5, .5])
        for name, value in kwargs.items():
            if name == "dynamics":
                dynamics = value
            elif name == "arg2":
                arg2 = value
        sc.dynamics = dynamics
        sc.addRobot(np.float32([[0, 0, 0], [0, 2/2, 0]]), arg2)
        
        # No leader
        sc.setADjMatrix(np.uint8([[0]]))
        # Set robot 0 as the leader.
        # sc.setADjMatrix(np.uint8([[0, 0, 0], [1, 0, 1], [1, 1, 0]]))
        
        # vrep related
        sc.initVrep()
        # Choose sensor type
        sc.SENSOR_TYPE = "None" # None, 2d, VPL16, kinect
        sc.objectNames = ['Pioneer_p3dx', 'Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
        
        if sc.SENSOR_TYPE == "None":
            sc.setVrepHandles(0, '')
        elif sc.SENSOR_TYPE == "2d":
            sc.objectNames.append('LaserScanner_2D_front')
            sc.objectNames.append('LaserScanner_2D_rear')
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
            sc.setVrepHandles(2, '#1')
        elif sc.SENSOR_TYPE == "VPL16":
            sc.objectNames.append('velodyneVPL_16') # _ptCloud
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
            sc.setVrepHandles(2, '#1')
        elif sc.SENSOR_TYPE == "kinect":
            sc.objectNames.append('kinect_depth')
            sc.objectNames.append('kinect_rgb')
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
            sc.setVrepHandles(2, '#1')
        
        #sc.renderScene(waitTime = 3000)
        tf = 30
        sp.plot(3, tf)
        while sc.simulate():
            #sc.renderScene(waitTime = int(sc.dt * 1000))
            #sc.showOccupancyMap(waitTime = int(sc.dt * 1000))
            
            #print("---------------------")
            #print("t = %.3f" % sc.t, "s")
            
            #sp.plot(0, tf)
            #sp.plot(1, tf)
            #sp.plot(2, tf) 
            sp.plot(3, tf)
            sp.plot(4, tf)
            sp.plot(5, tf)
            sp.plot(6, tf)
            if sc.t > tf:
                break
        sc.deallocate()
    except KeyboardInterrupt:
        x = input('Quit?(y/n)')
        sc.deallocate()
        if x == 'y' or x == 'Y':
            raise Exception('Aborted.')
        
    except:
        sc.deallocate()
        raise
    
    return None
# main
numRun = 6

for i in range(5, numRun):
    print('Run #: ', i, '...')
    sc = generateData(dynamics = 20, arg2 = i * np.float32([.3, .3]))
































