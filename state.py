# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 12:38:54 2017

@author: cz
"""

import sys
import math

class State():
    def __init__(self, x, y, theta, robot = None):
        self.x = x
        self.y = y
        self.vxp = 0 # vx prime, updated in controller
        self.vyp = 0 # vy prime, updated in controller
        self.theta = theta
        self.robot = robot
        
        
    def propagate(self, control):
        if self.robot == None:
            sys.exit("State: attribute robot is None")
        dt = self.robot.scene.dt
        if self.robot.dynamics <= 5:
            vx, vy = control()
            self.x += vx * dt
            self.y += vy * dt
            self.theta = 0
        elif self.robot.dynamics >= 5 and self.robot.dynamics < 10:
            vel, omega = control()
            self.x += vel * math.cos(self.theta) * dt
            self.y += vel * math.sin(self.theta) * dt
            self.theta += omega * dt
            print('omega: ' + str(omega))
        elif self.robot.dynamics >= 10:
            l = self.robot.l
            v1, v2 = control()
            self.x += math.cos(self.theta) * dt / 2 * (v1 + v2)
            self.y += math.sin(self.theta) * dt / 2 * (v1 + v2)
            self.theta += 1 / l * dt * (v2 - v1)
            
            
            
            
    def transform(self):
        # For feedback linearization
        c = self.robot.l / 2
        self.xp = self.x + c * math.cos(self.theta)
        self.yp = self.y + c * math.sin(self.theta)
        self.thetap = self.theta
        
        
        
        
        
        
        
        
            