# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:06:30 2017

@author: cz
"""

import cv2
import numpy as np
from robot import Robot
import matplotlib.pyplot as plt
import vrep
import math
import random

class Scene():
    def __init__(self):
        self.t = 0
        self.dt = 0.01
        
        # for plots
        self.ts = [] # timestamps
        self.tss = [] # timestamps (sparse)
        self.ydict = dict()
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
    def addRobot(self, arg, dynamics):
        robot = Robot(self)
        robot.index = len(self.robots)
        
        robot.xi.x = arg[0, 0]
        robot.xi.y = arg[0, 1]
        robot.xi.theta = arg[0, 2]
        robot.xid.x = arg[1, 0]
        robot.xid.y = arg[1, 1]
        robot.xid.theta = arg[1, 2]
        robot.xid0.x = arg[1, 0]
        robot.xid0.y = arg[1, 1]
        robot.xid0.theta = arg[1, 2]
        robot.dynamics = dynamics
        
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
        res, robotHandle = vrep.simxGetObjectHandle(
                self.clientID, handleNames[0] + handleNameSuffix, 
                vrep.simx_opmode_oneshot_wait)
        res, motorLeftHandle = vrep.simxGetObjectHandle(
                self.clientID, handleNames[1] + handleNameSuffix, 
                vrep.simx_opmode_oneshot_wait)
        res, motorRightHandle = vrep.simxGetObjectHandle(
                self.clientID, handleNames[2] + handleNameSuffix, 
                vrep.simx_opmode_oneshot_wait)
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
        boundaryFactor = 0.4
        if self.robots[0].dynamics == 12:
            # Generate random robot potiions whose center is at the origin
            alpha0 = (2/3 * random.random() + 1/6) * math.pi
            rho0 = boundaryFactor * self.xMax * random.random()
            x0 = rho0 * math.cos(alpha0)
            y0 = rho0 * math.sin(alpha0)
            theta0 = 2 * math.pi * random.random()
            self.robots[0].setPosition([x0, y0, theta0])
            
            alpha1 = (2/3 * random.random() - 1/2) * math.pi
            rho1 = boundaryFactor * self.xMax * random.random()
            x1 = rho1 * math.cos(alpha1)
            y1 = rho1 * math.sin(alpha1)
            theta1 = 2 * math.pi * random.random()
            self.robots[1].setPosition([x1, y1, theta1])
            
            x2 = -x0 - x1
            y2 = -y0 - y1
            theta2 = 2 * math.pi * random.random()
            self.robots[2].setPosition([x2, y2, theta2])
        
        
    def simulate(self):
        # vrep related
        '''
        cmd = input('Press <enter> key to step the simulation!')
        if cmd == 'q': # quit
            return False
        '''
        self.t += self.dt
        self.ts.append(self.t)
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
        for robot in self.robots:
            robot.draw(self.image, 1)
            robot.draw(self.image, 2)
        cv2.imshow('scene', self.image)
        cv2.waitKey(waitTime)
        
    def showOccupancyMap(self, waitTime = 25):
        wPix = self.robots[0].wPix
        hPix = self.robots[0].hPix
        N = len(self.robots)
        self.occupancyMap = np.ones((hPix, (wPix+1) * N), np.uint8) * 255
        x0 = 0
        for robot in self.robots:
            x1 = x0 + wPix
            self.occupancyMap[:, x0:x1] = robot.occupancyMap
            self.occupancyMap[:, x1:(x1+1)] = np.zeros((hPix, 1), np.uint8)
            x0 += wPix + 1
        #print('self.occupancyMap shape: ', self.occupancyMap.shape)
        resizeFactor = int(500/hPix)
        im = cv2.resize(self.occupancyMap, 
                        (self.occupancyMap.shape[1] * resizeFactor, 
                         self.occupancyMap.shape[0] * resizeFactor),
                        interpolation = cv2.INTER_NEAREST)
        cv2.imshow('Occupancy Map', im)
        cv2.waitKey(waitTime)
        
    def plot(self, type = 0, tf = 3):
        # If this is the first time this type of plot is drawn, initialize
        if type not in self.ydict.keys():
            self.ydict[type] = dict()
            self.ploted[type] = False 
            if type == 0 or type == 1:
                for i in range(len(self.robots)):
                    self.ydict[type][i] = []
            
        if self.ploted[type]:
            return # This type of plot is completed
        if type == 0:
            for i in range(len(self.robots)):
                x = self.robots[i].xi.x - self.robots[i].xid.x
                self.ydict[type][i].append(x)
            if self.t > tf:
                plt.figure(type)
                for i in range(len(self.robots)):
                    try:
                        plt.plot(self.ts, self.ydict[type][i], '-')
                    except:
                        print('type: ', type)
                        raise
                plt.xlabel('t (s)')
                plt.ylabel('x_i - x_di (m)')
                plt.show()
                self.ploted[type] = True
        elif type == 1:
            for i in range(len(self.robots)):
                x = self.robots[i].xi.y - self.robots[i].xid.y
                self.ydict[type][i].append(x)
            if self.t > tf:
                plt.figure(type)
                for i in range(len(self.robots)):
                    plt.plot(self.ts, self.ydict[type][i], '-')
                plt.xlabel('t (s)')
                plt.ylabel('y_i - y_di (m)')
                plt.show()
                self.ploted[type] = True
                
        elif type == 2: # Formation Error
            k = 0
            for i in range(len(self.robots)):
                for j in range(i + 1, len(self.robots)):
                    if self.adjMatrix[i, j] != 0:
                        # If this is the first time this type of plot is drawn
                        if type not in self.ydict[type].keys():
                            self.ydict[type][k] = []
                            #print('i = ', i, 'j = ', j)
                        xi = self.robots[i].xi.x
                        xj = self.robots[j].xi.x
                        yi = self.robots[i].xi.y
                        yj = self.robots[j].xi.y
                        d = pow(pow(xi - xj, 2) + pow(yi - yj, 2), 0.5)
                        self.ydict[type][k].append(d)
                        k += 1
            if self.t > tf:
                plt.figure(type)
                for k in range(len(self.ydict[type])):
                    plt.plot(self.ts, self.ydict[type][k], '-')
                plt.xlabel('t (s)')
                plt.ylabel('d_ij (m)')
                plt.show()
                self.ploted[type] = True
                
        elif type == 3:
            # Show formation
            # print('time: ', (self.t + 1e-5) % 1)
            if (self.t + 1e-5) % 3 < 2e-5:
                print("recording")
                self.tss.append(self.t)
                for i in range(len(self.robots)):
                    x = self.robots[i].xi.x
                    y = self.robots[i].xi.y
                    theta = self.robots[i].xi.theta*180/math.pi - 90 # convert to deg
                    if len(self.tss) == 1:
                        self.ydict[type][i] = np.array([[x, y, theta]])
                        if i == 0:
                            self.centerTrajS = np.array([[x, y]])
                        else:
                            self.centerTrajS += np.array([[x, y]])
                    else:
                        self.ydict[type][i] = np.append(self.ydict[type][i], 
                                  [[x, y, theta]], axis = 0)
                        if i == 0:
                            self.centerTrajS = np.append(self.centerTrajS, [[x, y]], axis = 0)
                        else:
                            self.centerTrajS[-1, :] += np.array([x, y])
                    #print(self.centerTrajS)
                self.centerTrajS[-1, :] /= len(self.robots)
                
            # Show Figure
            if self.t > tf:
                plt.figure(type)
                for i in range(len(self.robots)):
                    if i == 0:
                        c = (1, 0, 0)
                    elif i == 1:
                        c = (0, 1, 0)
                    elif i == 2:
                        c = (0, 0, 1)
                    for j in range(len(self.tss)):
                        plt.plot(self.ydict[type][i][j, 0], 
                                 self.ydict[type][i][j, 1], 
                                 marker=(3, 0, self.ydict[type][i][j, 2]),
                                 markersize=20, linestyle='None',
                                 color = c)
                
                l = len(self.robots)
                for i in range(len(self.robots)):
                    for j in range(len(self.tss)):
                        x1 = self.ydict[type][i][j, 0]
                        y1 = self.ydict[type][i][j, 1]
                        x2 = self.ydict[type][(i+1)%l][j, 0]
                        y2 = self.ydict[type][(i+1)%l][j, 1]
                        plt.plot([x1, x2], [y1, y2], '-', color = (0, 0, 0))
                        
                # Plot center trajectory
                for j in range(len(self.tss)):
                    plt.plot(self.centerTrajS[j, 0], 
                             self.centerTrajS[j, 1],
                             'o',
                             markersize=10, linestyle='None',
                             color = (0, 0, 0))
                
                plt.plot(self.centerTraj[:, 0], 
                         self.centerTraj[:, 1],
                         '-',
                         color = (0, 0, 0))
                
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                plt.axes().set_aspect('equal', 'datalim')
                plt.show()
                self.ploted[type] = True 
        
    
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
            
            
    
    
    
    
    
    