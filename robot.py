# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:08:18 2017

@author: cz
"""

import math
from state import State
import numpy as np
import cv2

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
            radius = 8
            omega = 1
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
        self.xi.propagate(self.control)
                
    def control(self):
        if self.dynamics == -1:
            # Leader Dynamics
            t = self.scene.t
            radius = 8
            omega = 1
            Kp = 5
            self.xid.x = radius * math.cos(omega * t)
            self.xid.y = radius * math.sin(omega * t)
            vx = -Kp * (self.xi.x - self.xid.x)
            vy = -Kp * (self.xi.y - self.xid.y)
            return vx, vy
        elif self.dynamics == 0:
            vx = 0
            vy = 0
            for j in range(len(self.scene.robots)):
                if self.scene.adjMatrix[self.index, j] == 0:
                    continue
                robot = self.scene.robots[j]
                vx += -((self.xi.x - robot.xi.x) - (self.xid.x - robot.xid.x))
                vy += -((self.xi.y - robot.xi.y) - (self.xid.y - robot.xid.y))
            return vx, vy
        elif self.dynamics == 1:
            Kp = 5
            vx = 0
            vy = 0
            for j in range(len(self.scene.robots)):
                if self.scene.adjMatrix[self.index, j] == 0:
                    continue
                robot = self.scene.robots[j]
                vx += -((self.xi.x - robot.xi.x) - (self.xid.x - robot.xid.x))
                vy += -((self.xi.y - robot.xi.y) - (self.xid.y - robot.xid.y))
            return Kp*vx, Kp*vy
        elif self.dynamics == 5:
            # For unicycle dynamics
            # First transform Cartesian coordinates to polar coordinates
            dx = self.xid.x - self.xi.x
            dy = self.xid.y - self.xi.y
            #print('dx: ' + str(dx) + '; dy:' + str(dy))
            rho = (dx ** 2 + dy ** 2) ** .5
            alpha = self.xi.theta - math.atan2(dy, dx)
            phi = self.xid.theta - self.xi.theta
            print('alpha: ' + str(alpha) + '; phi:' + str(phi))
            # Calculate control
            if rho > self.gamma:
                #vel = self.kRho * math.tanh(self.kV * rho)
                vel = self.kRho * rho
                omega = self.kAlpha * alpha + self.kPhi * phi
                self.reachedGoal = False
            else:
                vel = 0
                omega = 0
                self.reachedGoal = True
            return vel, omega
        elif self.dynamics == 6:
            # For unicycle dynamics
            i = self.index
            gammaXi = 0
            gammaYi = 0            
            for j in range(len(self.scene.robots)):
                robot = self.scene.robots[j]
                gammaXi += self.scene.Laplacian[i, j] * robot.xi.x
                gammaYi += self.scene.Laplacian[i, j] * robot.xi.y
                if self.scene.adjMatrix[self.index, j] != 0:
                    gammaXi += -(self.xid.x - robot.xid.x)
                    gammaYi += -(self.xid.y - robot.xid.y)
            thetaI = self.xi.theta
            thetaNHI = math.atan2(gammaYi, gammaXi)
            
            
            rho = 1
            if rho > self.gamma:
                vel = -(np.sign(gammaXi * math.cos(thetaI) +
                               gammaYi * math.sin(thetaI)) * 
                        (gammaXi ** 2 + gammaYi ** 2) ** 0.5)
                omega = -(thetaI - thetaNHI)
                self.reachedGoal = False
            else:
                vel = 0
                omega = 0
                self.reachedGoal = True
            return vel, omega
        elif self.dynamics == 4:
            # For single integrator dynamics.
            # Inputs are absolute positions
            Kp1 = 5
            Kp2 = 5
            vx = -Kp1 * (self.xi.x - self.xid.x)
            vy = -Kp1 * (self.xi.y - self.xid.y)
            for j in range(len(self.scene.robots)):
                if self.scene.adjMatrix[self.index, j] == 0:
                    continue
                robot = self.scene.robots[j]
                vx += -Kp2 * ((self.xi.x - robot.xi.x) - (self.xid.x - robot.xid.x))
                vy += -Kp2 * ((self.xi.y - robot.xi.y) - (self.xid.y - robot.xid.y))
            return vx, vy
        
        elif self.dynamics == 10:
            # For e-puk dynamics
            # Feedback linearization
            # v1: left wheel speed
            # v2: right wheel speed
            K1 = 1
            K2 = 1
            K3 = 0.5
            K4 = 0.5
            
            # velocity in transformed space
            vxp = self.xi.vxp
            vyp = self.xi.vyp
            
            for j in range(len(self.scene.robots)):
                if self.scene.adjMatrix[self.index, j] == 0:
                    continue
                robot = self.scene.robots[j] # neighbor
                #vxp += -K2 * (self.xi.xp - self.xid.xp)
                #vyp += -K2 * (self.xi.yp - self.xid.yp)
                vxp += -K3 * (self.xi.vxp - robot.xi.vxp)
                vyp += -K3 * (self.xi.vyp - robot.xi.vyp)
                vxp += -K4 * ((self.xi.xp - robot.xi.xp) - (self.xid.xp - robot.xid.xp))
                vyp += -K4 * ((self.xi.yp - robot.xi.yp) - (self.xid.yp - robot.xid.yp))
            
            self.xi.vxp = vxp
            self.xi.vyp = vyp
            
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
            
            return v1, v2

        elif self.dynamics == 11:
            # For e-puk dynamics
            # Feedback linearization
            # v1: left wheel speed
            # v2: right wheel speed
            K1 = 1
            K2 = 1
            K3 = 0.5
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
    

        
        