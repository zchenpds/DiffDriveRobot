# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:49:17 2018

@author: Zhuo Chen
"""
import numpy as np
import os
import queue

class Data():
    def __init__(self, robot):
        self.mode = -3
        self.q = queue.Queue()
        
        self.robot = robot
        pc = self.robot.pointCloud
        self.d = dict() # Will be equal to None after the scene is saved
        self.d['epi_starts'] = np.array([], dtype = np.bool)
        if self.mode == 0:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix), dtype = np.int8)
        elif self.mode == -1 or self.mode == -2:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix), dtype = np.int8)
            self.d['observations2'] = np.zeros((0, 1), dtype = np.float32)
        elif self.mode == -3:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix), dtype = np.int8)
            self.d['observations2'] = np.zeros((0, 2), dtype = np.float32)
        elif self.mode > 0:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix * 2), dtype = np.int8)
            self.d['observations2'] = np.zeros((0, 2), dtype = np.float32)
        self.d['observations1'] = np.zeros((0, pc.lenScanVector), dtype = np.float32)
        self.d['obs2'] = np.zeros((0, 11), dtype = np.float32)
        self.d['actions'] = np.zeros((0, 2), dtype = np.float32)
    def getObservation(self, mode):
        # This function can not run after scene has been saved as a pickle file
        if mode == 0:
            obs0 = self.robot.pointCloud.getObservation()
            act0 = self.robot.getV1V2()
            ret = (obs0, act0)
        elif mode > 0:
            obs0 = self.robot.pointCloud.getObservation()
            act0 = self.robot.getV1V2()
            #print(self.q.qsize())
            if self.q.qsize() == mode:
                obs1, act1 = self.q.get()
                obs = np.concatenate((obs0, obs1), axis = 1)
                act = act1
                ret = (obs, act)
            else:
                ret = (None, None)
            self.q.put((obs0, act0))
        elif mode < 0:
            obs0 = self.robot.pointCloud.getObservation()
            if mode == -1:
                leaderXi = self.robot.leader.xi
                vd = np.array([[(leaderXi.vx**2 + leaderXi.vy**2)**0.5]])
            elif mode == -2:
                followerXid = self.robot.xid
                vd = np.array([[(followerXid.vx**2 + followerXid.vy**2)**0.5]])
            elif mode == -3:
                followerXid = self.robot.xid
                vd = np.array([[followerXid.vx, followerXid.vy]])
            ret = (obs0, vd)
        
        return ret
    
    def add(self):
        # Add data corresponding to a particular point in time
        # This function can only run after the leader state is updated
        # This function can only run before self.robot is desctructed
        # This function can not run after scene has been saved as a pickle file
        observation, observation2 = self.getObservation(self.mode)
        if observation is None:
            return
        
        if len(self.d['epi_starts']) == 0:
            self.d['epi_starts'] = np.append(self.d['epi_starts'], True)
        else:
            self.d['epi_starts'] = np.append(self.d['epi_starts'], False)
        
        self.d['observations'] = np.append(self.d['observations'], observation, axis = 0) # option 1
        if self.mode != 0:
            self.d['observations2'] = np.append(self.d['observations2'], observation2, axis = 0)
            
        self.d['observations1'] = np.append(self.d['observations1'], 
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
        self.d['obs2'] = np.append(self.d['obs2'], obs2Data, axis = 0)
        self.d['actions'] = np.append(self.d['actions'], 
                  [[self.robot.v1Desired, self.robot.v2Desired]], axis = 0)
        
    def append(self, data2):
        # Append data collected in a run
        for key in self.d:
            self.d[key] =  np.append(self.d[key], data2.d[key], axis = 0)
    
    def store(self):
        i = self.robot.index
        directory = 'data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, 'data' + str(i))
        np.savez(path, **(self.d))
        message = "Training data of length {0:d} saved to " + path + ".npz"
        message = message.format(len(self.d['epi_starts']))
        self.robot.scene.log(message)
        
        