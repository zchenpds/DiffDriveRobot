# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:22:47 2018

@author: cz
"""
import matplotlib.pyplot as plt
import numpy as np
import math

class ScenePlot():
    def __init__(self, scene = None):
        self.sc = scene
        self.TYPE_TIME_ACTIONS = 6
    
    def plot(self, type = 0, tf = 0):
        # type 0: (t, x_i - x_id)
        # type 1: (t, y_i - y_id)
        # type 2: (t, e_i - e_id) (formation error)
        # type 3: (x_i, y_i)
        # type 4: (t, v1)
        # If this is the first time this type of plot is drawn, initialize
        if type not in self.sc.ydict.keys():
            self.sc.ydict[type] = dict()
            self.sc.ydict2[type] = dict() # for type 3 figure only
            self.sc.ploted[type] = False 
            if type == 0 or type == 1:
                for i in range(len(self.sc.robots)):
                    self.sc.ydict[type][i] = []
            
        if self.sc.ploted[type] and type != 6:
            pass#return # This type of plot is completed
        if type == 0:
            if not self.sc.ploted[type]:
                for i in range(len(self.sc.robots)):
                    x = self.sc.robots[i].xi.x - self.sc.robots[i].xid.x
                    self.sc.ydict[type][i].append(x)
            if self.sc.t > tf:
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    try:
                        plt.plot(self.sc.ts, self.sc.ydict[type][i], '-')
                    except:
                        print('type: ', type)
                        raise
                plt.xlabel('t (s)')
                plt.ylabel('x_i - x_di (m)')
                plt.show()
                self.sc.ploted[type] = True
        elif type == 1:
            if not self.sc.ploted[type]:
                for i in range(len(self.sc.robots)):
                    x = self.sc.robots[i].xi.y - self.sc.robots[i].xid.y
                    self.sc.ydict[type][i].append(x)
            if self.sc.t > tf:
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    plt.plot(self.sc.ts, self.sc.ydict[type][i], '-')
                plt.xlabel('t (s)')
                plt.ylabel('y_i - y_di (m)')
                plt.show()
                self.sc.ploted[type] = True
                
        elif type == 2: # Formation Error
            if not self.sc.ploted[type]:
                k = 0
                for i in range(len(self.sc.robots)):
                    for j in range(0, i):
                        if self.sc.adjMatrix[i, j] != 0:
                            # If this is the first time this type of plot is drawn
                            if k not in self.sc.ydict[type].keys():
                                self.sc.ydict[type][k] = []
                                # print(self.sc.ydict[type].keys())
                                # print('i = ', i, 'j = ', j)
                            xi = self.sc.robots[i].xi.x
                            xj = self.sc.robots[j].xi.x
                            yi = self.sc.robots[i].xi.y
                            yj = self.sc.robots[j].xi.y
                            xij = xi - xj
                            yij = yi - yj
                            d = (xij**2 + yij**2)**0.5
                            xi = self.sc.robots[i].xid.x
                            xj = self.sc.robots[j].xid.x
                            yi = self.sc.robots[i].xid.y
                            yj = self.sc.robots[j].xid.y
                            xijd = xi - xj
                            yijd = yi - yj
                            d0 = (xijd**2 + yijd**2)**0.5
                            self.sc.ydict[type][k].append(d - d0)
                            #print(self.sc.ydict[type][k])
                            k += 1
            if self.sc.t > tf:
                errors = self.sc.ydict[type]
                plt.figure(type)
                for k in range(len(errors)):
                    try:
                        plt.plot(self.sc.ts, errors[k], '-')
                    except:
                        print(k)
                        print(len(self.sc.ts))
                        print(len(errors[k]))
                plt.xlabel('t (s)')
                plt.ylabel('d_ij - d* (m)')
                plt.show()
                self.sc.ploted[type] = True
        
        elif type == 3: # Formation Error type 2
            if not self.sc.ploted[type]:
                for i in range(1, len(self.sc.robots)):
                    # If this is the first time this type of plot is drawn
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        # print(self.sc.ydict[type].keys())
                        # print('i = ', i, 'j = ', j)
                    xi = self.sc.robots[i].xi.x
                    yi = self.sc.robots[i].xi.y
                    xid = self.sc.robots[i].xid.x
                    yid = self.sc.robots[i].xid.y
                    error = ((xi - xid)**2 + (yi - yid)**2)**0.5
                    self.sc.ydict[type][i].append(error)
                    #print(self.sc.ydict[type][i])
            if self.sc.t > tf:
                errors = self.sc.ydict[type]
                plt.figure(type)
                for i in range(1, len(self.sc.robots)):
                    plt.plot(self.sc.ts, errors[i], '-')
                plt.xlabel('t (s)')
                plt.ylabel('((xi - xid)**2 + (yi - yid)**2)**0.5 (m)')
                plt.show()
                self.sc.ploted[type] = True
        
        elif type == 4:
            # Show formation
            if not self.sc.ploted[type]:
                # record individual trajectories
                for i in range(len(self.sc.robots)):
                    x = self.sc.robots[i].xi.x
                    y = self.sc.robots[i].xi.y
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict2[type][i] = np.array([[x, y]])
                    else: 
                        self.sc.ydict2[type][i] = np.append(self.sc.ydict2[type][i], 
                                   [[x, y]], axis = 0)
                        
                # print('time: ', (self.sc.t + 1e-5) % 1)
                if (self.sc.t + 1e-5) % 3 < 2e-5:
                    print("recording")
                    self.sc.tss.append(self.sc.t)
                    for i in range(len(self.sc.robots)):
                        x = self.sc.robots[i].xi.x
                        y = self.sc.robots[i].xi.y
                        theta = self.sc.robots[i].xi.theta*180/math.pi - 90 # convert to deg
                        if len(self.sc.tss) == 1:
                            self.sc.ydict[type][i] = np.array([[x, y, theta]])
                            if i == 0:
                                self.sc.centerTrajS = np.array([[x, y]])
                            else:
                                self.sc.centerTrajS += np.array([[x, y]])
                        else:
                            self.sc.ydict[type][i] = np.append(self.sc.ydict[type][i], 
                                      [[x, y, theta]], axis = 0)
                            if i == 0:
                                self.sc.centerTrajS = np.append(self.sc.centerTrajS, [[x, y]], axis = 0)
                            else:
                                self.sc.centerTrajS[-1, :] += np.array([x, y])
                        #print(self.sc.centerTrajS)
                    self.sc.centerTrajS[-1, :] /= len(self.sc.robots)
                
            # Show Figure
            if self.sc.t > tf:
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    c = self.sc.getRobotColor(i)
                    plt.plot(self.sc.ydict2[type][i][:, 0], 
                             self.sc.ydict2[type][i][:, 1], 
                             '-', color = c)
                    for j in range(len(self.sc.tss)):
                        plt.plot(self.sc.ydict[type][i][j, 0], 
                                 self.sc.ydict[type][i][j, 1], 
                                 marker=(3, 0, self.sc.ydict[type][i][j, 2]),
                                 markersize=20, linestyle='None',
                                 color = c)
                
                l = len(self.sc.robots)
                for i in range(len(self.sc.robots)):
                    for j in range(len(self.sc.tss)):
                        x1 = self.sc.ydict[type][i][j, 0]
                        y1 = self.sc.ydict[type][i][j, 1]
                        x2 = self.sc.ydict[type][(i+1)%l][j, 0]
                        y2 = self.sc.ydict[type][(i+1)%l][j, 1]
                        plt.plot([x1, x2], [y1, y2], '-', color = (0, 0, 0))
                        
                # Plot center trajectory
                for j in range(len(self.sc.tss)):
                    plt.plot(self.sc.centerTrajS[j, 0], 
                             self.sc.centerTrajS[j, 1],
                             'o',
                             markersize=10, linestyle='None',
                             color = (0, 0, 0))
                
                plt.plot(self.sc.centerTraj[:, 0], 
                         self.sc.centerTraj[:, 1],
                         '-',
                         color = (0, 0, 0))
                
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                plt.axes().set_aspect('equal', 'datalim')
                plt.show()
                self.sc.ploted[type] = True 
        
        elif type == 5:
            # Show speed
            if not self.sc.ploted[type]:
                for i in range(len(self.sc.robots)):
                    vDesired = (self.sc.robots[i].v1Desired + self.sc.robots[i].v2Desired)/2
                    if self.sc.vrepConnected ==  True:
                        vActual = self.sc.robots[i].vActual
                    else:
                        vActual = vDesired
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        self.sc.ydict2[type][i] = []
                    self.sc.ydict[type][i].append(vActual)
                    self.sc.ydict2[type][i].append(vDesired)
            if self.sc.t > tf:
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    c = self.sc.getRobotColor(i)
                    curve1, = plt.plot(self.sc.ts, self.sc.ydict[type][i], '-', 
                                      color = c, label = 'Actual')
                    curve2, = plt.plot(self.sc.ts, self.sc.ydict2[type][i], '--', 
                                      color = c, label = 'Desired')
                plt.legend(handles = [curve1, curve2])
                plt.xlabel('t (s)')
                plt.ylabel('v (m/s)')
                plt.show()
                self.sc.ploted[type] = True
        elif type == 6:
            # Show action
            if not self.sc.ploted[type]:
                for i in range(len(self.sc.robots)):
                    vDesired1 = self.sc.robots[i].v1Desired
                    vDesired2 = self.sc.robots[i].v2Desired
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        self.sc.ydict2[type][i] = []
                    self.sc.ydict[type][i].append(vDesired1)
                    self.sc.ydict2[type][i].append(vDesired2)
            if self.sc.t > tf:
                plt.figure(type)
                for i in range(1, len(self.sc.robots)): # Show only the follower
                #for i in range(len(self.sc.robots)): # Show both the follower and the leader
                    c = self.sc.getRobotColor(i)
                    curve1, = plt.plot(self.sc.ts, self.sc.ydict[type][i], ':', 
                                      color = c, label = 'Left Wheel Velocity')
                    curve2, = plt.plot(self.sc.ts, self.sc.ydict2[type][i], '--', 
                                      color = c, label = 'Right Wheel Velocity')
                plt.legend(handles = [curve1, curve2])
                plt.xlabel('t (s)')
                plt.ylabel('v (m/s)')
                plt.show()
                self.sc.ploted[type] = True
        elif type == 7:
            # Show angular velocity
            if not self.sc.ploted[type]:
                for i in range(len(self.sc.robots)):
                    omegaDesired = (self.sc.robots[i].v2Desired - 
                                    self.sc.robots[i].v1Desired) / self.sc.robots[i].l
                    if self.sc.vrepConnected ==  True:
                        omegaActual = self.sc.robots[i].omegaActual
                    else:
                        omegaActual = omegaDesired
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        self.sc.ydict2[type][i] = []
                    self.sc.ydict[type][i].append(omegaActual)
                    self.sc.ydict2[type][i].append(omegaDesired)
            if self.sc.t > tf:
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    c = self.sc.getRobotColor(i)
                    curve1, = plt.plot(self.sc.ts, self.sc.ydict[type][i], '-', 
                                      color = c, label = 'Actual')
                    curve2, = plt.plot(self.sc.ts, self.sc.ydict2[type][i], '--', 
                                      color = c, label = 'Desired')
                plt.legend(handles = [curve1, curve2])
                plt.xlabel('t (s)')
                plt.ylabel('omega (rad/s)')
                plt.show()
                self.sc.ploted[type] = True        
        
        elif type == 8:
            # Show Euler angles
            if not self.sc.ploted[type]:
                if self.sc.vrepConnected ==  False:
                    return
                for i in range(len(self.sc.robots)):
                    alpha = self.sc.robots[i].xi.alpha / math.pi * 180
                    beta = self.sc.robots[i].xi.beta / math.pi * 180
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        self.sc.ydict2[type][i] = []
                    self.sc.ydict[type][i].append(alpha)
                    self.sc.ydict2[type][i].append(beta)
            if self.sc.t > tf:
                if len(self.sc.ts) != len(self.sc.ydict[type][0]):
                    return
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    c = self.sc.getRobotColor(i)
                    curve1, = plt.plot(self.sc.ts, self.sc.ydict[type][i], '-', 
                                      color = c, label = 'alpha')
                    curve2, = plt.plot(self.sc.ts, self.sc.ydict2[type][i], '--', 
                                      color = c, label = 'beta')
                plt.legend(handles = [curve1, curve2])
                plt.xlabel('t (s)')
                plt.ylabel('angles (deg)')
                plt.show()
                self.sc.ploted[type] = True     
                
        elif type == 9:
            # Show observation1
            for i in range(len(self.sc.robots)):
                plt.figure()
                c = self.sc.getRobotColor(i)
                pc = self.sc.robots[i].pointCloud
                dist = pc.scanVector
                angle = pc.scanAngle
                for j in range(pc.lenScanVector):
                    x = -dist[0, j] * math.cos(angle[j])
                    y = -dist[0, j] * math.sin(angle[j])
                    plt.plot([0, x], [0, y], '-', color = c)
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                plt.axes().set_aspect('equal')
                plt.show()
            self.sc.ploted[type] = True