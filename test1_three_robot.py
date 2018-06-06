# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:09:34 2017
This test file is dependent on vrep.
To run this file, please open vrep file scene/scene_double.ttt first
@author: cz
"""

from scene import Scene
from robot import VrepError
from sceneplot import ScenePlot
# from robot import Robot
import numpy as np
import math
import random
# from data import Data
#from DeepFCL import DeepFCL

#fcl = DeepFCL(50, 50, 2, 1)

def initRef(sc, i):
    message = ""
    if sc.dynamics == 5:
        sc.xid.vxp = 0#.2
        sc.xid.vyp = 0#.3
    elif sc.dynamics == sc.DYNAMICS_MODEL_BASED_DISTANCE_GOAL:
        g = 4.0 
        goalList = [[g, g], [-g, g], [g, -g], [-g, -g]]
        #sc.xid.x, sc.xid.y = random.choice(goalList)
        sc.xid.x, sc.xid.y = goalList[i]
        sc.xid.vRef = 0.7
        message = "Goal: ({0:.3f}, {1:.3f})"
        message = message.format(sc.xid.x, sc.xid.y, sc.xid.vRef)
        sc.xid.theta = 0
        sc.xid.sDot = 0
        sc.xid.thetaDot = 0
    elif sc.dynamics == sc.DYNAMICS_MODEL_BASED_DISTANCE_REFVEL:
        # set desired velocity vector
        sc.xid.vRefMag = 0.5
        sc.xid.vRefAng = 2 * math.pi * random.random()
        sc.xid.theta = 0
        sc.xid.sDot = 0
        sc.xid.thetaDot = 0
        # scale desired formation separation
        #alphaList = [1.0, 1.5, 2.0]
        alphaList = [1.0]
        alpha = random.choice(alphaList)
        sc.scaleDesiredFormation(alpha)
        message = "vRefMag: {0:.3f}, vRefAng: {1:.3f}, alpha: {2:.3f}"
        message = message.format(sc.xid.vRefMag, sc.xid.vRefAng, alpha)
    if len(message) > 0:
        sc.log(message)
        print(message)

def plot(sp, tf):
    #sp.plot(0, tf)
    sp.plot(2, tf) # Formation Separation
    if sp.sc.dynamics == 5:
        pass#sp.plot(24, tf) # epsilon
    if sp.sc.dynamics == 16:
        sp.plot(23, tf) # distance from goal
    elif sp.sc.dynamics == 14:
        sp.plot(22, tf) # distances from goals
    sp.plot(4, tf)
    #sp.plot(5, tf)
    sp.plot(6, tf)
    
def generateData(i):
    sc = Scene(fileName = __file__, recordData = False, runNum = i)
    sp = ScenePlot(sc)
    sp.saveEnabled = True # save plots?
    #sc.occupancyMapType = sc.OCCUPANCY_MAP_THREE_CHANNEL
    sc.occupancyMapType = sc.OCCUPANCY_MAP_BINARY
    sc.dynamics = 5 # robot dynamics
    sc.errorType = 0
    try:
        sc.addRobot(np.float32([[-.0, 0, 0], [0.0, 0.0, 0.0]]), role = sc.ROLE_PEER)
        sc.addRobot(np.float32([[-3, 4, 0], [1.0, 0.0, 0.0]]), role = sc.ROLE_PEER)
        sc.addRobot(np.float32([[2, 1, 0], [0.0, 1.732, 0.0]]), role = sc.ROLE_PEER)
#==============================================================================
#         sc.addRobot(np.float32([[1, 3, 0], [0, -1, 0]]), 
#                     dynamics = sc.DYNAMICS_LEARNED, 
#                     learnedController = fcl.test)
#==============================================================================
        
        # No leader
        sc.setADjMatrix(np.uint8([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        # Set robot 0 as the leader.
        
        
        #sc.renderScene(waitTime = 3000)
        tf = 15 # must be greater than 1
        errorCheckerEnabled = False
        initRef(sc, i)
        #sc.resetPosition(2) # Random initial position
        sc.robots[0].setPosition([.0, .0, .0])
        sc.robots[1].setPosition([-2.0, 0.001, 0.0])
        sc.robots[2].setPosition([2.0, 0.0, 0.0])
        
        #sc.robots[0].setPosition([.0, .0, .0])
        #sc.robots[1].setPosition([-3.0, 4.0, 0.0])
        #sc.robots[2].setPosition([2.0, 1.0, 0.0])
        
        # Fixed initial position
        #sc.robots[0].setPosition([0.0, 0.0, math.pi/2]) 
        #sc.robots[1].setPosition([-2.2, -1.0, 0.3])
        sp.plot(4, tf)
        while sc.simulate():
            sc.renderScene(waitTime = int(sc.dt * 1000), mode = 1)
            #sc.showOccupancyMap(waitTime = int(sc.dt * 1000))
            
            #print("---------------------")
            #print("t = %.3f" % sc.t, "s")
            #print(sc.robots[2].xid0.y)
            if sc.t > 1:
                maxAbsError = sc.getMaxFormationError()
                if maxAbsError < 0.01 and errorCheckerEnabled:
                    #tf = sc.t - 0.01
                    # set for how many seconds after convergence the simulator shall run
                    tExtra = 30
                    #tf = sc.t + tExtra
                    errorCheckerEnabled = False
                    print('Ending in ', str(tExtra), ' seconds...')
            
            plot(sp, tf)
            if sc.t > tf:
                message = "maxAbsError = {0:.3f} m".format(maxAbsError)
                sc.log(message)
                print(message)
                break
            
            
    except KeyboardInterrupt:
        x = input('Quit?(y/n)')
        if x == 'y' or x == 'Y':
            tf = sc.t - 0.01
            plot(sp, tf)
            raise Exception('Aborted.')
    
    except VrepError as err:
        sc.log(err.message)
        print(err.message)
        return None
    except:
        raise
    finally:
        sc.deallocate()
    
    if True: #maxAbsError < 0.01:
        return sc
    else:
        return None



# main
numRun = 1


for i in range(0, numRun):
    print('Run #: ', i, '...')
    # First episode
    sc0 = generateData(i)































