# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:08:18 2017

@author: cz
"""

import math
from state import State
import numpy as np
import cv2
import vrep

class Robot():
    def __init__(self, scene):
        self.scene = scene
        self.dynamics = 1
        
        # dynamics parameters
        self.l = 0.331
        
        # state
        self.xi = State(0, 0, 0, self)
        self.xid = State(3, 0, 0, self)
        self.xid0 = State(3, 0, math.pi/4, self)
        self.reachedGoal = False
        # Control parameters
        self.kRho = 1
        self.kAlpha = 6
        self.kPhi = -1
        self.kV = 3.8
        self.gamma = 0.15
        
        # For visualization of occupancy map
        self.wPix = 50
        self.hPix = 50
        self.xMax = 5
        self.yMax = 5
        self.occupancyMap = np.ones((self.hPix, self.wPix), np.uint8) * 255

    def propagateDesired(self):
        if self.dynamics == 4 or self.dynamics == 11:
            t = self.scene.t
            radius = 2
            omega = 0.3
            theta0 = math.atan2(self.xid0.y, self.xid0.x)
            rho0 = (self.xid0.x ** 2 + self.xid0.y ** 2) ** 0.5
            self.xid.x = (radius * math.cos(omega * t) +
                          rho0 * math.cos(omega * t + theta0))
            self.xid.y = (radius * math.sin(omega * t) +
                          rho0 * math.sin(omega * t + theta0))
            self.xid.vx = -(radius * omega * math.sin(omega * t) +
                            rho0 * omega * math.sin(omega * t + theta0))
            self.xid.vy = (radius * omega * math.cos(omega * t) +
                           rho0 * omega * math.cos(omega * t + theta0))
            self.xid.theta = math.atan2(self.xid.vy, self.xid.vx)
            #self.xid.omega = omega
            
            c = self.l/2
            self.xid.vxp = self.xid.vx - c * math.sin(self.xid.theta) * omega
            self.xid.vyp = self.xid.vy + c * math.cos(self.xid.theta) * omega
        elif self.dynamics == 12:
            self.xid.x = self.xid0.x
            self.xid.y = self.xid0.y
            self.xid.vx = 0
            self.xid.vy = 0
            self.xid.theta = 0
            #self.xid.omega = omega
            
            c = self.l/2
            self.xid.vxp = 0
            self.xid.vyp = 0
            
    def precompute(self):
        if self.dynamics >= 10:
            self.xi.transform()
            self.xid.transform()
        
    def propagate(self):
        if self.scene.vrepConnected == False:
            self.xi.propagate(self.control)
        else:
            omega1, omega2 = self.control()
            vrep.simxSetJointTargetVelocity(self.scene.clientID, 
                                            self.motorLeftHandle, 
                                            omega1, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(self.scene.clientID, 
                                            self.motorRightHandle, 
                                            omega2, vrep.simx_opmode_oneshot)
    def control(self):
        if self.dynamics == 11 or self.dynamics == 12:
            # For e-puk dynamics
            # Feedback linearization
            # v1: left wheel speed
            # v2: right wheel speed
            K1 = 1
            K2 = 1
            #K3 = 1
            K4 = 0.5
            
            # velocity in transformed space
            vxp = 0
            vyp = 0
            
            for j in range(len(self.scene.robots)):
                if self.scene.adjMatrix[self.index, j] == 0:
                    continue
                robot = self.scene.robots[j] # neighbor
                vxp += -K4 * ((self.xi.xp - robot.xi.xp) - (self.xid.xp - robot.xid.xp))
                vyp += -K4 * ((self.xi.yp - robot.xi.yp) - (self.xid.yp - robot.xid.yp))
            
            vxp += -K1 * (self.xi.xp - self.xid.xp)
            vyp += -K1 * (self.xi.yp - self.xid.yp)
            
            vxp += K2 * self.xid.vxp
            vyp += K2 * self.xid.vyp
            
            kk = 1
            theta = self.xi.theta
            M11 = kk * math.sin(theta) + math.cos(theta)
            M12 =-kk * math.cos(theta) + math.sin(theta)
            M21 =-kk * math.sin(theta) + math.cos(theta)
            M22 = kk * math.cos(theta) + math.sin(theta)
            
            v1 = M11 * vxp + M12 * vyp
            v2 = M21 * vxp + M22 * vyp
            
            #v1 = 0.3
            #v2 = 0.3
            
            #print("v1 = %.3f" % v1, "m/s, v2 = %.3f" % v2)
            
            vm = 1.5 # wheel's max linear speed in m/s
            # Find the factor for converting linear speed to angular speed
            if math.fabs(v2) >= math.fabs(v1) and math.fabs(v2) > vm:
                alpha = vm / math.fabs(v2)
            elif math.fabs(v2) < math.fabs(v1) and math.fabs(v1) > vm:
                alpha = vm / math.fabs(v1)
            else:
                alpha = 1
                
            v1 = alpha * v1
            v2 = alpha * v2
            
            if self.scene.vrepConnected:
                omega1 = v1 * 10
                omega2 = v2 * 10
                # return angular speeds of the two wheels
                return omega1, omega2
            else:
                # return linear speeds of the two wheels
                return v1, v2
        
    def draw(self, image, drawType):
        if drawType == 1:
            xi = self.xi
            color = (0, 0, 255)
        elif drawType == 2:
            xi = self.xid
            color = (0, 255, 0)
        r = self.l/2
        rPix = round(r * self.scene.m2pix())
        dx = -r * math.sin(xi.theta)
        dy = r * math.cos(xi.theta)
        p1 = np.float32([[xi.x + dx, xi.y + dy]])
        p2 = np.float32([[xi.x - dx, xi.y - dy]])
        p0 = np.float32([[xi.x, xi.y]])
        p3 = np.float32([[xi.x + dy/2, xi.y - dx/2]])
        p1Pix = self.scene.m2pix(p1)
        p2Pix = self.scene.m2pix(p2)
        p0Pix = self.scene.m2pix(p0)
        p3Pix = self.scene.m2pix(p3)
        if self.dynamics <= 1 or self.dynamics == 4:
            cv2.circle(image, tuple(p0Pix[0]), rPix, color)
        else:
            cv2.line(image, tuple(p1Pix[0]), tuple(p2Pix[0]), color)
            cv2.line(image, tuple(p0Pix[0]), tuple(p3Pix[0]), color)
    
    def setPosition(self, stateVector = None):
        # stateVector = [x, y, theta]
        if self.scene.vrepConnected == False:
            return
        z0 = 0.1587
        if stateVector == None:
            x0 = self.xi.x
            y0 = self.xi.y
            theta0 = self.xi.theta
        elif len(stateVector) == 3:
            x0 = stateVector[0]
            y0 = stateVector[1]
            theta0 = stateVector[2]
            
        vrep.simxSetObjectPosition(self.scene.clientID, self.robotHandle, -1, 
                [x0, y0, z0], vrep.simx_opmode_oneshot)
        vrep.simxSetObjectOrientation(self.scene.clientID, self.robotHandle, -1, 
                [0, 0, theta0], vrep.simx_opmode_oneshot)

    def readSensorData(self):
        if self.scene.vrepConnected == False:
            return
        if "readSensorData_firstCall" not in self.__dict__: 
            self.readSensorData_firstCall = True
        else:
            self.readSensorData_firstCall = False
        
        # Read robot states
        res, pos = vrep.simxGetObjectPosition(self.scene.clientID, 
                                              self.robotHandle, -1, 
                                              vrep.simx_opmode_blocking)
        res, ori = vrep.simxGetObjectOrientation(self.scene.clientID, 
                                              self.robotHandle, -1, 
                                              vrep.simx_opmode_blocking)
        res, vel, omega = vrep.simxGetObjectVelocity(self.scene.clientID,
                                                     self.robotHandle,
                                                     vrep.simx_opmode_blocking)
        #print("Linear speed: %.3f" % (vel[0]**2 + vel[1]**2)**0.5, 
        #      "m/s. Angular speed: %.3f" % omega[2])
        #print("pos: %.2f" % pos[0], ", %.2f" % pos[1])
        #print("ori: %.2f" % ori[0], ", %.2f" % ori[1], ", %.2f" % ori[2])
        self.xi.x = pos[0]
        self.xi.y = pos[1]
        self.xi.theta = ori[2]
        
        # Read laser/vision sensor data
        if self.scene.SENSOR_TYPE == "2d_":
            # self.laserFrontHandle
            # self.laserRearHandle
            
            if self.readSensorData_firstCall:
                opmode = vrep.simx_opmode_streaming
            else:
                opmode = vrep.simx_opmode_buffer
            laserFront_points = vrep.simxGetStringSignal(
                    self.scene.clientID, self.laserFrontName + '_signal', opmode)
            print(self.laserFrontName + '_signal: ', len(laserFront_points[1]))
            laserRear_points = vrep.simxGetStringSignal(
                    self.scene.clientID, self.laserRearName + '_signal', opmode)
            print(self.laserRearName + '_signal: ', len(laserRear_points[1]))
        elif self.scene.SENSOR_TYPE == "2d":
            # self.laserFrontHandle
            # self.laserRearHandle
            
            if self.readSensorData_firstCall:
                opmode = vrep.simx_opmode_streaming
            else:
                opmode = vrep.simx_opmode_buffer
            laserFront_points = vrep.simxCallScriptFunction(
                    self.scene.clientID, self.laserFrontName, 1, 
                    'getLaserData_function', [], [], [], 'abc', 
                    vrep.simx_opmode_blocking)
            #print(self.laserFrontName + '_signal: ', laserFront_points[2])
            #print(self.laserFrontName + ' length: ', len(laserFront_points[2]))
            laserRear_points = vrep.simxCallScriptFunction(
                    self.scene.clientID, self.laserRearName, 1, 
                    'getLaserData_function', [], [], [], 'abc', 
                    vrep.simx_opmode_blocking)
            #print(self.laserRearName + '_signal: ', laserRear_points[2])
            #print(self.laserRearName + ' length: ', len(laserRear_points[2]))
            
            # Parse data
            self.pointCloud = np.zeros((0, 2), np.float32)
            for i in range(0, len(laserFront_points[2]), 3):
                x = laserFront_points[2][i + 1]
                y = laserFront_points[2][i + 2]
                if (x > self.xMax or x < -self.xMax or 
                    y > self.yMax or y < -self.yMax):
                    continue
                self.pointCloud = np.vstack((self.pointCloud, 
                                                  np.float32([x, y])))
            for i in range(0, len(laserRear_points[2]), 3):
                x = laserRear_points[2][i + 1]
                y = laserRear_points[2][i + 2]
                if (x > self.xMax or x < -self.xMax or 
                    y > self.yMax or y < -self.yMax):
                    continue
                self.pointCloud = np.vstack((self.pointCloud, 
                                                  np.float32([-x, -y])))
            pointCloudPix = self.m2pix(self.pointCloud)
            #print('pointCloudPix ', pointCloudPix)
            self.occupancyMap = np.ones((self.hPix, self.wPix), np.uint8) * 255
            r = int(self.l/2*self.m2pix())
            cv2.circle(self.occupancyMap, 
                       (int(self.hPix/2), int(self.wPix/2)), r, 180, -1)
            for i in range(pointCloudPix.shape[0]):
                cv2.circle(self.occupancyMap, 
                           (pointCloudPix[i, 0], pointCloudPix[i, 1]), 1, 0, -1)
        elif self.scene.SENSOR_TYPE == "VPL16":
            # self.pointCloudHandle
            velodyne_points = vrep.simxCallScriptFunction(
                    self.scene.clientID, self.pointCloudName, 1, 
                    'getVelodyneData_function', [], [], [], 'abc', 
                    vrep.simx_opmode_blocking)
            print(len(velodyne_points[2]))
            #print(velodyne_points[2])
            
            # Parse data
            if 'VPL16_counter' not in self.__dict__:
                self.VPL16_counter = 0
            # reset the counter every fourth time
            if self.VPL16_counter == 4:
                self.VPL16_counter = 0
            if self.VPL16_counter == 0:
                # Reset point cloud and occupancy map
                self.pointCloud = np.zeros((0, 2), np.float32)
            print('VPL16_counter = ', self.VPL16_counter)
            for i in range(0, len(velodyne_points[2]), 3):
                x = velodyne_points[2][i]
                z = velodyne_points[2][i + 1]
                y = velodyne_points[2][i + 2]
                MIN = 0.28
                if (x > self.xMax or x < -self.xMax or 
                    y > self.yMax or y < -self.yMax or z < - 0.3):
                    continue
                elif (x < MIN and y < MIN and x > -MIN and y > -MIN):
                    continue
                self.pointCloud = np.vstack((self.pointCloud, 
                                                  np.float32([x, y])))
            if self.VPL16_counter == 3:
                self.occupancyMap = np.ones((self.hPix, self.wPix), np.uint8) * 255
                r = int(self.l/2*self.m2pix())
                #cv2.circle(self.occupancyMap, 
                #           (int(self.hPix/2), int(self.wPix/2)), r, 180, -1)
                pointCloudPix = self.m2pix(self.pointCloud)
                for i in range(pointCloudPix.shape[0]):
                    self.occupancyMap[(pointCloudPix[i, 0], pointCloudPix[i, 1])] = 0
            self.VPL16_counter += 1
            
        elif self.scene.SENSOR_TYPE == "kinect":
            pass
        else:
            return
        
    def m2pix(self, p = None):
        if p is None: # if p is None
            return (self.wPix / self.xMax / 2)
        xPix = ((self.xMax - p[:, 0]) * (self.wPix / self.xMax / 2))
        yPix = ((self.yMax + p[:, 1]) * (self.hPix / self.yMax / 2))
        #print('x, y: ' +str(np.uint16([[x, y]])))
        xyPix = np.uint16([xPix, yPix])
        return xyPix.transpose()
        