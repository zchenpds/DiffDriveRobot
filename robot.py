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
        

    def propagateDesired(self):
        if self.dynamics == 4 or self.dynamics >= 10:
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
            
    def precompute(self):
        if self.dynamics >= 10:
            self.xi.transform()
            self.xid.transform()
        
    def propagate(self):
        if self.scene.vrepConnected == False:
            self.xi.propagate(self.control)
        else:
            res, pos = vrep.simxGetObjectPosition(self.scene.clientID, 
                                                  self.robotHandle, -1, 
                                                  vrep.simx_opmode_blocking)
            res, ori = vrep.simxGetObjectOrientation(self.scene.clientID, 
                                                  self.robotHandle, -1, 
                                                  vrep.simx_opmode_blocking)
            res, vel, omega = vrep.simxGetObjectVelocity(self.scene.clientID,
                                                         self.robotHandle,
                                                         vrep.simx_opmode_blocking)
            print("Linear speed: %.3f" % (vel[0]**2 + vel[1]**2)**0.5, 
                  "m/s. Angular speed: %.3f" % omega[2])
            #print("pos: %.2f" % pos[0], ", %.2f" % pos[1])
            #print("ori: %.2f" % ori[0], ", %.2f" % ori[1], ", %.2f" % ori[2])
            self.xi.x = pos[0]
            self.xi.y = pos[1]
            self.xi.theta = ori[2]
            v1, v2 = self.control()
            vrep.simxSetJointTargetVelocity(self.scene.clientID, 
                                            self.motorLeftHandle, 
                                            v1, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(self.scene.clientID, 
                                            self.motorRightHandle, 
                                            v2, vrep.simx_opmode_oneshot)
    def control(self):
        if self.dynamics == 11:
            # For e-puk dynamics
            # Feedback linearization
            # v1: left wheel speed
            # v2: right wheel speed
            K1 = 1
            K2 = 1
            K3 = 1
            K4 = 0.2
            
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
            
            print("v1 = %.3f" % v1, "m/s, v2 = %.3f" % v2)
            
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
    

        
        