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
import datetime
import os
from state import State

class Scene():
    def __init__(self, fileName = "Untitled", recordData = False, runNum = 0):
        self.t = 0
        self.dt = 0.01
        
        # formation reference link
        self.xid = State(0.0, 0.0, math.pi / 2)
        self.xi = State(0.0, 0.0, math.pi / 2)
        self.alpha = 1 # desired formation scale
        
        # for plots
        self.ts = [] # timestamps
        self.tss = [] # timestamps (sparse)
        self.ydict = dict()
        self.ydict2 = dict()
        self.ploted = dict()
        
        # For visualization
        self.wPix = 600
        self.hPix = 600
        self.xMax = 8
        self.yMax = 8
        self.image = np.zeros((self.hPix, self.wPix, 3), np.uint8)
        if USE_CV2:
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter('output.avi', self.fourcc, 20.0, (self.wPix, self.hPix))
            self.frameCounter = 0
            
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
        self.DYNAMICS_INTEGRATOR_MODEL = 5
        self.DYNAMICS_MODEL_BASED_CICULAR = 11
        self.DYNAMICS_MODEL_BASED_STABILIZER = 12
        self.DYNAMICS_MODEL_BASED_LINEAR = 13
        self.DYNAMICS_MODEL_BASED_LINEAR_GOAL = 14
        self.DYNAMICS_MODEL_BASED_DISTANCE_GOAL = 16
        self.DYNAMICS_MODEL_BASED_DISTANCE_REFVEL = 17
        self.DYNAMICS_MODEL_BASED_DISTANCE2_REFVEL = 18
        self.DYNAMICS_LEARNED = 30
        
        # follower does not have knowledge of absolute position
        self.ROLE_LEADER = 0
        self.ROLE_FOLLOWER = 1
        self.ROLE_PEER = 2
        
        self.errorType = 0
        self.logPriorityMax = 1 # Messages with lower priorities are not logged
        self.logFileName = os.path.splitext(fileName)[0] + ".log"
        self.runNum = runNum
        self.log('A new scene is created for run #' + str(runNum))
        
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
        
        message = ""
        if robot.role == 0:
            message += "Leader"
        elif robot.role == 1:
            message += "Follower"
        elif robot.role == 2:
            message += "Peer"
        else:
            message += "Type-unkonwn"
        message += " robot #" + str(robot.index) + " using "
        if learnedController is None:
            message += "a model-based controller"
        else:
            message += "a leanrned controller"
        message += " is added to the scene"
        self.log(message)
    
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
            print ("Failed connecting to remote API server")
            raise Exception("Failed connecting to remote API server")
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
        
    def resetPosition(self, radius = 2):
        if radius is None:
            for i in range(0, len(self.robots)):
                self.robots[i].setPosition(None)
            return
        
        MIN_DISTANCE = 1
        if self.dynamics == 5:
            xbar = 0
            ybar = 0
            for i in range(0, len(self.robots)):
                while True:
                    minDij = float("inf")
                    #alpha1 = math.pi * (-2/3*i - 1/3* random.random()) # limited
                    alpha1 = math.pi * (2 * random.random()) # arbitrary
                    rho1 = radius * random.random()
                    x1 = rho1 * math.cos(alpha1)
                    y1 = rho1 * math.sin(alpha1)
                    theta1 = 0#2 * math.pi * random.random()
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
                xbar += x1
                ybar += y1
            self.xi.x = xbar / len(self.robots)
            self.xi.y = ybar / len(self.robots)
            self.xid.dpbarx = self.xi.x - self.xid.x
            self.xid.dpbary = self.xi.y - self.xid.y
        elif self.dynamics >= 16 and self.dynamics <= 18:
            xbar = 0
            ybar = 0
            for i in range(0, len(self.robots)):
                while True:
                    minDij = float("inf")
                    #alpha1 = math.pi * (-2/3*i - 1/3* random.random()) # limited
                    alpha1 = math.pi * (2 * random.random()) # arbitrary
                    rho1 = radius * random.random()
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
                xbar += x1
                ybar += y1
            self.xi.x = xbar / len(self.robots)
            self.xi.y = ybar / len(self.robots)
            self.xid.dpbarx = self.xi.x - self.xid.x
            self.xid.dpbary = self.xi.y - self.xid.y
            
        #input('One moment.')
        # End of resetPosition()     
        
        
        
        
        
    def scaleDesiredFormation(self, alpha):
        self.alpha = alpha
        for robot in self.robots:
            robot.xid0.x *= alpha
            robot.xid0.y *= alpha
            robot.xid.x *= alpha
            robot.xid.y *= alpha

    def propagateXid(self):
        t = self.t
        dt = self.dt
        sDot = 0
        thetaDot = 0        
        if self.robots[0].dynamics == 13:
            t1 = 1
            speed = self.referenceSpeed
            omega = self.referenceOmega
            if t < t1:
                sDot = t / t1 * speed
                thetaDot = t / t1 * omega
            else:
                sDot = speed
                thetaDot = omega
            self.xid.x += sDot * dt * math.cos(self.xid.theta)
            self.xid.y += sDot * dt * math.sin(self.xid.theta)
            self.xid.theta += thetaDot * dt
            self.xid.sDot = sDot
            self.xid.thetaDot = thetaDot
        elif self.robots[0].dynamics == 14:
            # do nothing because xid is time-invariant
            pass
        elif self.robots[0].dynamics == 16:
            xbar = 0
            ybar = 0
            for robot in self.robots:
                xbar += robot.xi.x
                ybar += robot.xi.y
            self.xi.x = xbar / len(self.robots)
            self.xi.y = ybar / len(self.robots)
            self.xid.dpbarx = self.xi.x - self.xid.x
            self.xid.dpbary = self.xi.y - self.xid.y
            #print('dpbarx: ', self.xid.dpbarx, ', dpbary: ', self.xid.dpbary)
        elif self.dynamics == 17 or self.dynamics == 18:
            # self.xid.vRefMag
            # self.xid.vRefAng
            omega = 0
            self.xid.dpbarx = -self.xid.vRefMag * math.cos(self.xid.vRefAng + self.t * omega)
            self.xid.dpbary = -self.xid.vRefMag * math.sin(self.xid.vRefAng + self.t * omega)
            #print('dpbarx: ', self.xid.dpbarx, ', dpbary: ', self.xid.dpbary)
        
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
         
    def renderScene(self, timestep = -1, waitTime = 25, mode = 0):
        if USE_CV2 == False:
            return
        self.image = np.zeros((self.hPix, self.wPix, 3), np.uint8)
        for robot in self.robots:
            robot.draw(self.image, 1)
            #robot.draw(self.image, 2)
        if mode == 0:
            cv2.imshow('scene', self.image)
            cv2.waitKey(waitTime)
        elif mode == 1:
            if self.frameCounter % 5 == 0:
                self.out.write(self.image)
            self.frameCounter += 1
            
    
    def getRobotColor(self, i, brightness = 0.7, reverse = False):
        
        if i == 0:
            c = (brightness, 0, 0)
        elif i == 1:
            c = (0, brightness, 0)
        elif i == 2:
            c = (0, 0, brightness)
        elif i == 3:
            c = (0, brightness, brightness)
        else:
            c = (brightness, 0, brightness)
        if reverse == True:
            return c[::-1]
        else:
            return c
    
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
        self.log("Scene is destructed")
        if USE_CV2 == True:
            cv2.destroyAllWindows() # Add this to fix the window freezing bug
            self.out.release()
        # vrep related
        if self.vrepConnected:
            self.vrepConnected = False
            # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
            #vrep.simxGetPingTime(self.clientID)
            # Stop simulation:
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
            # Now close the connection to V-REP:
            vrep.simxFinish(self.clientID)
            
            
    def log(self, message, priority=1):
        if priority <= self.logPriorityMax:
            with open(self.logFileName, "a+" ) as f:
                prefix = ("[" + str(datetime.datetime.now()) + "]"
                            + " [run #{0:03d}]"
                            + " [sim time: {1:.3f} s] ")
                prefix = prefix.format(self.runNum, self.t)
                f.write(prefix + message + '\n')
    
    
    
    
