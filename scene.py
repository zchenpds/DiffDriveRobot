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

class Scene():
    def __init__(self):
        self.t = 0
        self.dt = 0.01
        
        # for plots
        self.ts = []
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
        else:
            self.vrepConnected = False
            print ('Failed connecting to remote API server')
            raise
        self.dt = 0.05
    
    def setVrepHandles(self, robotIndex, handleNames, handleNameSuffix = ""):
        if self.vrepConnected == False:
            return False
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
            robot.propagateDesired()
            robot.propagate()
            if robot.reachedGoal:
                countReachedGoal += 1
        if self.vrepConnected:
            vrep.simxSynchronousTrigger(self.clientID);
        if countReachedGoal == len(self.robots):
            return False
        else:
            return True
         
    def renderScene(self, timestep = -1, waitTime = 25):
        for robot in self.robots:
            robot.draw(self.image, 1)
            robot.draw(self.image, 2)
        cv2.imshow('scene', self.image)
        cv2.waitKey(waitTime)
        
    def plot(self, type = 0, tf = 3):
        if type not in self.ydict.keys():
            self.ydict[type] = dict()
            self.ploted[type] = False
            for i in range(len(self.robots)):
                self.ydict[type][i] = []
        if self.ploted[type]:
            return
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
        elif type == 2: #???
            for i in range(len(self.robots)):
                x = self.robots[i].xi.x - self.robots[i].xid.x
                self.ydict[type][i].append(x)
            if self.t > tf:
                plt.figure(type)
                for i in range(len(self.robots)):
                    plt.plot(self.ts, self.ydict[type][i], '-')
                plt.xlabel('t (s)')
                plt.ylabel('x_i - x_di')
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
            
            
    
    
    
    
    
    