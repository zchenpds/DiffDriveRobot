# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:09:34 2017
This test file is dependent on vrep.
To run this file, please open vrep file scene/scene_double.ttt first
@author: cz
"""

from scene import Scene
# from robot import Robot
import numpy as np
# from data import Data
from DeepFCL import DeepFCL

fcl = DeepFCL(50, 50, 2, 1)

def generateData():
    sc = Scene(recordData = True)
    #sc.occupancyMapType = sc.OCCUPANCY_MAP_THREE_CHANNEL
    sc.occupancyMapType = sc.OCCUPANCY_MAP_BINARY
    sc.dynamics = sc.DYNAMICS_MODEL_BASED_LINEAR # robot dynamics
    try:
        sc.addRobot(np.float32([[-2, 0, 0], [0.0, 0.0, 0.0]]), role = sc.ROLE_LEADER)
        sc.addRobot(np.float32([[1, 3, 0], [-1.0, 0.0, 0.0]]), role = sc.ROLE_FOLLOWER)
#==============================================================================
#         sc.addRobot(np.float32([[1, 3, 0], [0, -1, 0]]), 
#                     dynamics = sc.DYNAMICS_LEARNED, 
#                     learnedController = fcl.test)
#==============================================================================
        
        # No leader
        sc.setADjMatrix(np.uint8([[0, 0], [1, 0]]))
        # Set robot 0 as the leader.
        
        # vrep related
        sc.initVrep()
        # Choose sensor type
        sc.SENSOR_TYPE = "VPL16" # None, 2d, VPL16, kinect
        #sc.SENSOR_TYPE = "None" # None, 2d, VPL16, kinect
        sc.objectNames = ['Pioneer_p3dx', 'Pioneer_p3dx_leftMotor', 'Pioneer_p3dx_rightMotor']
        
        if sc.SENSOR_TYPE == "None":
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
        elif sc.SENSOR_TYPE == "2d":
            sc.objectNames.append('LaserScanner_2D_front')
            sc.objectNames.append('LaserScanner_2D_rear')
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
        elif sc.SENSOR_TYPE == "VPL16":
            sc.objectNames.append('velodyneVPL_16') # _ptCloud
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
        elif sc.SENSOR_TYPE == "kinect":
            sc.objectNames.append('kinect_depth')
            sc.objectNames.append('kinect_rgb')
            sc.setVrepHandles(0, '')
            sc.setVrepHandles(1, '#0')
        
        #sc.renderScene(waitTime = 3000)
        tf = 30 # must be greater than 1
        sc.resetPosition()
        sc.plot(3, tf)
        while sc.simulate():
            #sc.renderScene(waitTime = int(sc.dt * 1000))
            sc.showOccupancyMap(waitTime = int(sc.dt * 1000))
            
            #print("---------------------")
            #print("t = %.3f" % sc.t, "s")
            
            if sc.t > 1:
                maxAbsError = sc.getMaxFormationError()
                if maxAbsError < 0.01:
                    tf = sc.t - 0.01
            
            #sc.plot(0, tf)
            sc.plot(2, tf)
            #sc.plot(1, tf) 
            sc.plot(3, tf)
            sc.plot(4, tf)
            sc.plot(5, tf)
            sc.plot(6, tf)
            if sc.t > tf:
                print('maxAbsError = ', maxAbsError)
                break
            
            
                
        
            #print('robot 0: ', sc.robots[0].xi.x, ', ', sc.robots[0].xi.y, ', ', sc.robots[0].xi.theta)
            #print('robot 1: ', sc.robots[1].xi.x, ', ', sc.robots[1].xi.y, ', ', sc.robots[1].xi.theta)
            #print('robot 2: ', sc.robots[2].xi.x, ', ', sc.robots[2].xi.y, ', ', sc.robots[2].xi.theta)
            #print('y01: ' + str(sc.robots[1].xi.y - sc.robots[0].xi.y))
            #print('x02: ' + str(sc.robots[2].xi.x - sc.robots[0].xi.x))
        sc.deallocate()
    except KeyboardInterrupt:
        x = input('Quit?(y/n)')
        sc.deallocate()
        if x == 'y' or x == 'Y':
            raise Exception('Aborted.')
        
    except:
        sc.deallocate()
        raise
    
    
    
    if True: #maxAbsError < 0.01:
        return sc
    else:
        return None
# main
numRun = 3
dataList = []


for i in range(0, numRun):
    print('Run #: ', i, '...')
    # First episode
    sc = generateData()
    if sc is not None:
        # if the list is empty
        if not dataList:
            for robot in sc.robots:
                dataList.append(robot.data)
        else:
            for j in range(len(sc.robots)):
                dataList[j].append(sc.robots[j].data)
        

for j in range(len(sc.robots)):
    dataList[j].store()

































