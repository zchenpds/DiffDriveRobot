# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:06:30 2017

@author: cz
"""
try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False

import numpy as np
from robot import Robot
#import matplotlib.pyplot as plt
import vrep
import math
import random
from state import State


REFERENCE_SPEED = 0.3
REFERENCE_THETA_DOT = 0

class Scene():
    def __init__(self, recordData = False):
        self.t = 0
        self.dt = 0.01
        
        # formation reference link
        self.xid = State(0.0, 0.0, math.pi / 2)
        
        # for plots
        self.ts = [] # timestamps
        self.tss = [] # timestamps (sparse)
        self.ydict = dict()
        self.ydict2 = dict()
        self.ploted = dict()
        
        # For visualization
        self.wPix = 600
        self.hPix = 600
        self.xMax = 5
        self.yMax = 5
        self.image = np.zeros((self.hPix, self.wPix, 3), np.uint8)
        
        self.robots = []
        self.adjMatrix = None
        self.Laplacian = None
         
        # vrep related
        self.vrepConnected = False
        #self.vrepSimStarted = False
        self.SENSOR_TYPE = "None"
        self.objectNames = []
        self.recordData = recordData
        
        self.occupancyMapType = None
        self.OCCUPANCY_MAP_BINARY = 0
        # 1 for 3-channel: mean height, height variance, visibility
        self.OCCUPANCY_MAP_THREE_CHANNEL = 1
        
        # CONSTANTS
        self.dynamics = 11
        self.DYNAMICS_MODEL_BASED_CICULAR = 11
        self.DYNAMICS_MODEL_BASED_STABILIZER = 12
        self.DYNAMICS_MODEL_BASED_LINEAR = 13
        self.DYNAMICS_LEARNED = 30
        
        # follower does not have knowledge of absolute position
        self.ROLE_LEADER = 0
        self.ROLE_FOLLOWER = 1
        
        self.errorType = 1
        
    def addRobot(self, arg, arg2 = np.float32([.5, .5]), 
                 role = 1, learnedController = None):
        robot = Robot(self)
        robot.index = len(self.robots)
        
        robot.role = role
        robot.learnedController = learnedController
        robot.xi.x = arg[0, 0]
        robot.xi.y = arg[0, 1]
        robot.xi.theta = arg[0, 2]
        robot.xid.x = arg[1, 0]
        robot.xid.y = arg[1, 1]
        robot.xid.theta = arg[1, 2]
        robot.xid0.x = arg[1, 0]
        robot.xid0.y = arg[1, 1]
        robot.xid0.theta = arg[1, 2]
        robot.dynamics = self.dynamics
        
        if self.dynamics >=20 and self.dynamics <= 25:
            robot.arg2 = arg2
        
        if robot.role == self.ROLE_LEADER:
            robot.recordData = False # Leader data is not recorded
        else:
            robot.recordData = self.recordData
        
        self.robots.append(robot)
    
    def setADjMatrix(self, adjMatrix):
        self.adjMatrix = adjMatrix
        self.Laplacian = np.diag(np.sum(self.adjMatrix, axis = 1))
    
    def initVrep(self):
        print ('Program started')
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
        if self.clientID!=-1:
            self.vrepConnected = True
            print('Connected to remote API server')
             # enable the synchronous mode on the client:
            vrep.simxSynchronous(self.clientID, True)
            # start the simulation:
            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
            # Laser Scanner Initialization
            #if self.SENSOR_TYPE == "2d":
                
        else:
            self.vrepConnected = False
            print ('Failed connecting to remote API server')
            raise
        self.dt = 0.05
    
    def setVrepHandles(self, robotIndex, handleNameSuffix = ""):
        if self.vrepConnected == False:
            return False
        handleNames = self.objectNames
        res1, robotHandle = vrep.simxGetObjectHandle(
                self.clientID, handleNames[0] + handleNameSuffix, 
                vrep.simx_opmode_oneshot_wait)
        
        res2, motorLeftHandle = vrep.simxGetObjectHandle(
                self.clientID, handleNames[1] + handleNameSuffix, 
                vrep.simx_opmode_oneshot_wait)
        res3, motorRightHandle = vrep.simxGetObjectHandle(
                self.clientID, handleNames[2] + handleNameSuffix, 
                vrep.simx_opmode_oneshot_wait)
        print("Vrep res: ", res1, res2, res3)
        self.robots[robotIndex].robotHandle = robotHandle
        self.robots[robotIndex].motorLeftHandle = motorLeftHandle
        self.robots[robotIndex].motorRightHandle = motorRightHandle
        #print(self.robots[robotIndex].robotHandle)
        
        if self.SENSOR_TYPE == "None":
            pass
        elif self.SENSOR_TYPE == "2d":
            res, laserFrontHandle = vrep.simxGetObjectHandle(
                    self.clientID, handleNames[3] + handleNameSuffix, 
                    vrep.simx_opmode_oneshot_wait)
            print('2D Laser (front) Initilization:', 'Successful' if not res else 'error')
            res, laserRearHandle = vrep.simxGetObjectHandle(
                    self.clientID, handleNames[4] + handleNameSuffix, 
                    vrep.simx_opmode_oneshot_wait)
            print('2D Laser (rear) Initilization:', 'Successful' if not res else 'error')
            self.robots[robotIndex].laserFrontHandle = laserFrontHandle
            self.robots[robotIndex].laserRearHandle = laserRearHandle
            self.robots[robotIndex].laserFrontName = handleNames[3] + handleNameSuffix
            self.robots[robotIndex].laserRearName = handleNames[4] + handleNameSuffix
        elif self.SENSOR_TYPE == "VPL16":
            res, pointCloudHandle = vrep.simxGetObjectHandle(
                    self.clientID, handleNames[3] + handleNameSuffix, 
                    vrep.simx_opmode_oneshot_wait)
            print('Point Cloud Initilization:', 'Successful' if not res else 'error')
            self.robots[robotIndex].pointCloudHandle = pointCloudHandle
            self.robots[robotIndex].pointCloudName = handleNames[3] + handleNameSuffix
        elif self.SENSOR_TYPE == "kinect":
            res, kinectDepthHandle = vrep.simxGetObjectHandle(
                    self.clientID, handleNames[3] + handleNameSuffix, 
                    vrep.simx_opmode_oneshot_wait)
            print('Kinect Depth Initilization: ', 'Successful' if not res else 'error')
            res, kinectRgbHandle = vrep.simxGetObjectHandle(
                    self.clientID, handleNames[4] + handleNameSuffix, 
                    vrep.simx_opmode_oneshot_wait)
            print('Kinect RGB Initilization: ', 'Successful' if not res else 'error')
            self.robots[robotIndex].kinectDepthHandle = kinectDepthHandle
            self.robots[robotIndex].kinectRgbHandle = kinectRgbHandle
            self.robots[robotIndex].kinectDepthName = handleNames[3] + handleNameSuffix
            self.robots[robotIndex].kinectRgbName = handleNames[4] + handleNameSuffix
            
        #self.robots[robotIndex].setPosition()
        self.robots[robotIndex].readSensorData()
        
    def resetPosition(self):
        boundaryFactor = 0.7
        MIN_DISTANCE = 1
        if self.robots[0].dynamics == 11:
            for i in range(0, len(self.robots)):
                while True:
                    minDij = 100
                    alpha1 = 2 * math.pi * random.random()
                    rho1 = boundaryFactor * self.xMax * random.random()
                    x1 = rho1 * math.cos(alpha1)
                    y1 = rho1 * math.sin(alpha1)
                    theta1 = 2 * math.pi * random.random()
                    for j in range(0, i):
                        dij = pow( pow(x1 - self.robots[j].xi.x, 2) + 
                                   pow(y1 - self.robots[j].xi.y, 2), 0.5)
                        # print('j = ', j, '( %.3f' % self.robots[j].xi.x, ', %.3f'%self.robots[j].xi.y, '), ', 'dij = ', dij)
                        if dij < minDij:
                            minDij = dij # find the smallest dij for all j
                    print('Min distance: ', minDij, 'from robot #', i, 'to other robots.')
                    
                    # if the smallest dij is greater than allowed,
                    if i==0 or minDij >= MIN_DISTANCE:
                        self.robots[i].setPosition([x1, y1, theta1])
                        break # i++
                        
        elif self.robots[0].dynamics == 12:
            self.robots[0].setPosition([0.0, 1.0, 0.0])
            for i in range(1, len(self.robots)):
                while True:
                    minDij = 100
                    alpha1 = 2 * math.pi * random.random()
                    rho1 = boundaryFactor * self.xMax * random.random()
                    x1 = rho1 * math.cos(alpha1)
                    y1 = rho1 * math.sin(alpha1)
                    theta1 = 2 * math.pi * random.random()
                    for j in range(0, i):
                        dij = pow( pow(x1 - self.robots[j].xi.x, 2) + 
                                   pow(y1 - self.robots[j].xi.y, 2), 0.5)
                        # print('j = ', j, '( %.3f' % self.robots[j].xi.x, ', %.3f'%self.robots[j].xi.y, '), ', 'dij = ', dij)
                        if dij < minDij:
                            minDij = dij # find the smallest dij for all j
                    print('Min distance: ', minDij, 'from robot #', i, 'to other robots.')
                    
                    # if the smallest dij is greater than allowed,
                    if minDij >= MIN_DISTANCE:
                        self.robots[i].setPosition([x1, y1, theta1])
                        break # i++
                        
        elif self.robots[0].dynamics == 13:
            self.robots[0].setPosition([0.0, 0.0, math.pi/2])
            for i in range(1, len(self.robots)):
                while True:
                    minDij = 100
                    alpha1 = math.pi * (1 + random.random())
                    rho1 = boundaryFactor * self.xMax * random.random()
                    x1 = rho1 * math.cos(alpha1)
                    y1 = rho1 * math.sin(alpha1)
                    theta1 = 2 * math.pi * random.random()
                    for j in range(0, i):
                        dij = ((x1 - self.robots[j].xi.x)**2 + 
                               (y1 - self.robots[j].xi.y)**2)**0.5
                        # print('j = ', j, '( %.3f' % self.robots[j].xi.x, ', %.3f'%self.robots[j].xi.y, '), ', 'dij = ', dij)
                        if dij < minDij:
                            minDij = dij # find the smallest dij for all j
                    print('Min distance: ', minDij, 'from robot #', i, 'to other robots.')
                    
                    # if the smallest dij is greater than allowed,
                    if minDij >= MIN_DISTANCE:
                        self.robots[i].setPosition([x1, y1, theta1])
                        break # i++
        #input('One moment.')
        # End of resetPosition()


    def propagateXid(self):
        t = self.t
        dt = self.dt
        sDot = 0
        thetaDot = 0
        if self.dynamics == 13:
            t1 = 1
            speed = REFERENCE_SPEED
            if t < t1:
                sDot = t / t1 * speed
                thetaDot = t / t1 * REFERENCE_THETA_DOT
            else:
                sDot = speed
                thetaDot = REFERENCE_THETA_DOT
        self.xid.x += sDot * dt * math.cos(self.xid.theta)
        self.xid.y += sDot * dt * math.sin(self.xid.theta)
        self.xid.theta += thetaDot * dt
        self.xid.sDot = sDot
        self.xid.thetaDot = thetaDot
        
    def simulate(self):
        # vrep related
        '''
        cmd = input('Press <enter> key to step the simulation!')
        if cmd == 'q': # quit
            return False
        '''
        self.t += self.dt
        self.ts.append(self.t)
        self.propagateXid()
        countReachedGoal = 0
        for robot in self.robots:
            robot.precompute()
        for robot in self.robots:
            robot.readSensorData()
            robot.propagateDesired()
            robot.propagate()
            if robot.reachedGoal:
                countReachedGoal += 1
        self.calcCOG()
        
        if self.vrepConnected:
            vrep.simxSynchronousTrigger(self.clientID);
        if countReachedGoal == len(self.robots):
            return False
        else:
            return True
        
    def calcCOG(self):
        # Calculate Center Of Gravity
        for i in range(len(self.robots)):
            x = self.robots[i].xi.x
            y = self.robots[i].xi.y
            if len(self.ts) == 1:
                if i == 0:
                    self.centerTraj = np.array([[x, y]])
                else:
                    self.centerTraj += np.array([[x, y]])
            else:
                if i == 0:
                    self.centerTraj = np.append(self.centerTraj, [[x, y]], axis = 0)
                else:
                    #print('size', self.centerTraj.shape)
                    self.centerTraj[-1, :] += np.array([x, y])
            #print(self.centerTraj)
        self.centerTraj[-1, :] /= len(self.robots)
         
    def renderScene(self, timestep = -1, waitTime = 25):
        if USE_CV2 == False:
            return
        for robot in self.robots:
            robot.draw(self.image, 1)
            robot.draw(self.image, 2)
        cv2.imshow('scene', self.image)
        cv2.waitKey(waitTime)
        
    def showOccupancyMap(self, waitTime = 25):
        if USE_CV2 == False:
            return
        pc = self.robots[0].pointCloud
        wPix = pc.wPix
        hPix = pc.hPix
        N = len(self.robots)
        resizeFactor = int(500/hPix)
        if self.occupancyMapType == self.OCCUPANCY_MAP_BINARY:
            self.occupancyMap = np.ones((hPix, (wPix+1) * N), np.uint8) * 255
            x0 = 0
            for robot in self.robots:
                x1 = x0 + wPix
                self.occupancyMap[:, x0:x1] = robot.pointCloud.occupancyMap
                self.occupancyMap[:, x1:(x1+1)] = np.zeros((hPix, 1), np.uint8)
                x0 += wPix + 1
            #print('self.occupancyMap shape: ', self.occupancyMap.shape)
            
            im = cv2.resize(self.occupancyMap, 
                            (self.occupancyMap.shape[1] * resizeFactor, 
                             self.occupancyMap.shape[0] * resizeFactor),
                            interpolation = cv2.INTER_NEAREST)
            cv2.imshow('Occupancy Map', im)
        elif self.occupancyMap == self.OCCUPANCY_MAP_THREE_CHANNEL:
            self.occupancyMap = np.zeros((hPix, (wPix+1) * N, 3), np.uint8)
            x0 = 0
            for robot in self.robots:
                x1 = x0 + wPix
                self.occupancyMap[:, x0:x1, :] = robot.pointCloud.occupancyMap
                self.occupancyMap[:, x1:(x1+1), :] = np.ones((hPix, 1, 3), np.uint8) * 255
                x0 += wPix + 1
            #print('self.occupancyMap shape: ', self.occupancyMap.shape)
            im = cv2.resize(self.occupancyMap, 
                            (self.occupancyMap.shape[1] * resizeFactor, 
                             self.occupancyMap.shape[0] * resizeFactor),
                            interpolation = cv2.INTER_NEAREST)
            cv2.imshow('Occupancy Map', im)
        cv2.waitKey(waitTime)
        
    def getRobotColor(self, i):
        if i == 0:
            c = (1, 0, 0)
        elif i == 1:
            c = (0, 1, 0)
        elif i == 2:
            c = (0, 0, 1)
        return c
    
    def getMaxFormationError(self):
        if 2 not in self.ydict.keys():
            raise Exception('Plot type 2 must be drawn in order to get formation error!')
        if self.errorType == 0:
            errors = self.ydict[2]
        else:
            errors = self.ydict[3]
        # check max formation error
        maxAbsError = 0
        for key in errors:
            absError = abs(errors[key][-1])
            if absError > maxAbsError:
                maxAbsError = absError
        return maxAbsError
    
    def m2pix(self, p = None):
        if p is None: # if p is None
            return (self.wPix / self.xMax / 2)
        x, y = tuple(p[0])
        #print('x = ' + str(x) + ', y = ' + str(y))
        xPix = int((x + self.xMax) * (self.wPix / self.xMax / 2))
        yPix = int((self.yMax - y) * (self.hPix / self.yMax / 2))
        #print('x, y: ' +str(np.uint16([[x, y]])))
        if (xPix < self.wPix and xPix >= 0 and
            yPix < self.hPix and yPix >= 0):
            return np.uint16([[xPix, yPix]])
        else:
            return None
    
    
    def deallocate(self):
        if USE_CV2 == True:
            cv2.destroyAllWindows() # Add this to fix the window freezing bug
        
        # vrep related
        if self.vrepConnected:
            self.vrepConnected = False
            # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
            #vrep.simxGetPingTime(self.clientID)
            # Stop simulation:
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
            # Now close the connection to V-REP:
            vrep.simxFinish(self.clientID)
            
            
    
    
    
    
    
    