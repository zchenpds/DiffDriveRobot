# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:49:17 2018

@author: Zhuo Chen
"""
import numpy as np
import os

class Data():
    def __init__(self, robot):
        self.robot = robot
        pc = self.robot.pointCloud
        self.epi_starts = np.array([], dtype = np.bool)
        self.observations = np.zeros((0, pc.hPix * pc.wPix), dtype = np.int8)
        self.observations1 = np.zeros((0, pc.lenScanVector), dtype = np.float32)
        self.obs2 = np.zeros((0, 11), dtype = np.float32)
        self.actions = np.zeros((0, 2), dtype = np.float32)
    
    def add(self):
        # Add data corresponding to a particular point in time
        # This function can only run after the leader state is updated
        # This function can only run before self.robot is desctructed
        if len(self.epi_starts) == 0:
            self.epi_starts = np.append(self.epi_starts, True)
        else:
            self.epi_starts = np.append(self.epi_starts, False)
        self.observations = np.append(self.observations, 
                              self.robot.pointCloud.getObservation(), 
                              axis = 0) # option 1
        self.observations1 = np.append(self.observations1, 
                              self.robot.pointCloud.scanVector, axis = 0) # option 2
        followerXi = self.robot.xi
        leaderXi = self.robot.leader.xi
        followerXid = self.robot.xid
        obs2Data = [[leaderXi.x, leaderXi.y, # 1, 2
                    followerXi.x, followerXi.y,  # 3, 4
                    leaderXi.x - followerXi.x, leaderXi.y - followerXi.y, # 5, 6
                    followerXid.vx, followerXid.vy, # 7, 8
                    leaderXi.theta, followerXi.theta, # 9, 10
                    leaderXi.theta - followerXi.theta # 11
                    ]]
        self.obs2 = np.append(self.obs2, obs2Data, axis = 0)
        self.actions = np.append(self.actions, [[self.robot.v1Desired, 
                                                 self.robot.v2Desired]], axis = 0)
    
    def append(self, data2):
        # Append data collected in a run
        self.epi_starts = np.append(self.epi_starts, data2.epi_starts)
        self.observations = np.append(self.observations, data2.observations, axis = 0) # option 1
        self.observations1 = np.append(self.observations1, data2.observations1, axis = 0) # option 2
        self.obs2 = np.append(self.obs2, data2.obs2, axis = 0)
        self.actions = np.append(self.actions, data2.actions, axis = 0)
    
    def store(self):
        i = self.robot.index
        directory = 'data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savez(os.path.join(directory, 'data' + str(i)), 
                 epi_starts = self.epi_starts,
                 observations = self.observations,
                 observations1 = self.observations1,
                 obs2 = self.obs2, 
                 actions = self.actions)
        