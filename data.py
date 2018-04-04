# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:49:17 2018

@author: Zhuo Chen
"""
import numpy as np
import os
import queue
import math

class Data():
    def __init__(self, robot):
        self.mode = -10
        self.q = queue.Queue()
        
        self.robot = robot
        pc = self.robot.pointCloud
        self.d = dict() # Will become None after the scene is saved
        self.d['epi_starts'] = np.array([], dtype = np.bool)
        if self.mode == 0:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix), dtype = np.int8)
        elif self.mode == -1 or self.mode == -2:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix), dtype = np.int8)
            self.d['observations2'] = np.zeros((0, 1), dtype = np.float32)
        elif self.mode == -3 or self.mode == -4:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix), dtype = np.int8)
            self.d['observations2'] = np.zeros((0, 2), dtype = np.float32)
        elif self.mode == -10:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix), dtype = np.int8)
            self.d['observations2'] = np.zeros((0, 3), dtype = np.float32)
        elif self.mode > 0:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix * 2), dtype = np.int8)
            self.d['observations2'] = np.zeros((0, 2), dtype = np.float32)
        self.d['observations1'] = np.zeros((0, pc.lenScanVector), dtype = np.float32)
        if self.robot.scene.dynamics == 13:
            self.d['obs2'] = np.zeros((0, 17), dtype = np.float32)
        elif self.robot.scene.dynamics == 14:
            self.d['obs2'] = np.zeros((0, 7), dtype = np.float32)
        else:
            raise Exception("Undefined robot dynamics for data recording", self.robot.dynamics)
        self.d['actions'] = np.zeros((0, 2), dtype = np.float32)
    def getObservation(self, mode):
        # This function can not run after scene has been saved as a pickle file
        if mode == 0:
            obs0 = self.robot.pointCloud.getObservation()
            act0 = self.robot.getV1V2()
            ret = (obs0, act0)
        elif mode > 0:
            obs0 = self.robot.pointCloud.getObservation()
            xi0 = self.robot.xi
            #print(self.q.qsize())
            if self.q.qsize() == mode:
                xi1 = self.q.get()
                obs = np.concatenate(xi1, axis = 1)
                ret = (obs, None)
            else:
                ret = (None, None)
            self.q.put(xi0)
        elif mode < 0:
            obs0 = self.robot.pointCloud.getObservation()
            if mode == -1: # 1x1 Leader's actual speed
                vLeader = self.robot.leader.getV1V2()
                state = np.array([[0.5*(vLeader[0, 0] + vLeader[0, 1])]])
            elif mode == -2: # 1x1 Follower's reference speed
                followerXid = self.robot.xid
                state = np.array([[(followerXid.vx**2 + followerXid.vy**2)**0.5]])
            elif mode == -3: # 1x2 Follower's reference speed
                followerXid = self.robot.xid
                state = np.array([[followerXid.vx, followerXid.vy]])
            elif mode == -4: # 1x2 Velocities of the leader's two wheels
                state = self.robot.leader.getV1V2()
            elif mode == -10: # Peer's state
                peer = self.robot
                phii = math.atan2(peer.xid0.y, peer.xid0.x)
                rhoi = (peer.xid0.x ** 2 + peer.xid0.y ** 2) ** 0.5
                thetai = peer.xi.theta
                state = np.array([[peer.xid.vRef, rhoi, phii - thetai]])
            ret = (obs0, state)
        
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
        
        if self.robot.scene.dynamics == 13:
            followerXi = self.robot.xi
            leaderXi = self.robot.leader.xi
            followerXid = self.robot.xid
            vLeader = self.robot.leader.getV1V2()
            obs2Data = [[leaderXi.x, leaderXi.y, # 1, 2
                        followerXi.x, followerXi.y,  # 3, 4
                        leaderXi.x - followerXi.x, leaderXi.y - followerXi.y, # 5, 6
                        followerXid.vx, followerXid.vy, # 7, 8
                        leaderXi.theta, followerXi.theta, # 9, 10
                        leaderXi.theta - followerXi.theta, # 11
                        0.5*(vLeader[0, 0] + vLeader[0, 1]), #12 : mode = -1
                        (followerXid.vx**2 + followerXid.vy**2)**0.5, #13: mode = -2
                        followerXid.vx, followerXid.vy, #14, 15: mode = -3
                        vLeader[0, 0], vLeader[0, 1] #16, 17: mode = -3
                        ]]
        elif self.robot.scene.dynamics == 14:
            peer = self.robot
            phii = math.atan2(peer.xid0.y, peer.xid0.x)
            rhoi = (peer.xid0.x ** 2 + peer.xid0.y ** 2) ** 0.5
            thetai = peer.xi.theta
            obs2Data = [[peer.xid.vRef, rhoi, phii - thetai, # 1, 2, 3: mode = -10
                         peer.xid.x - peer.xi.x, peer.xid.y - peer.xi.y,  # 4, 5
                         peer.xid.x, peer.xid.y]] # 6, 7
                         
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
        
        
