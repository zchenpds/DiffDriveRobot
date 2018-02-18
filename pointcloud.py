# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:08:39 2018

@author: cz
"""
import numpy as np
import math

class PointCloud():
    def __init__(self, robot):
        self.robot = robot
        self.data = []
        self.dataCropped = []
        
        # For visualization of laser scan vector (option 2)
        self.lenScanVector = 50
        self.maxRange = 5
        self.scanVector = np.ones((1, self.lenScanVector), np.float32) * self.maxRange
        self.scanAngle = ((np.asarray(range(self.lenScanVector)) + 0.5)  * 
                          (2 * math.pi / self.lenScanVector))
        
        # For visualization of occupancy map (option 1)
        self.wPix = 50
        self.hPix = 50
        self.xMax = 5
        self.yMax = 5
        if self.robot.scene.occupancyMapType == self.robot.scene.OCCUPANCY_MAP_BINARY:
            self.occupancyMap = np.ones((self.hPix, self.wPix), np.uint8) * 255
        elif self.robot.scene.occupancyMapType == self.robot.scene.OCCUPANCY_MAP_THREE_CHANNEL:
            self.occupancyMap = np.zeros((self.hPix, self.wPix, 3), np.uint8)
        
        
    def clearData(self):
        self.data = []
        self.dataCropped = []
        
    def addRawData(self, rawData):
        newData = []
        for i in range(0, len(rawData), 3):
            x = rawData[i]
            z = rawData[i + 1]
            y = rawData[i + 2]
            newData.append(np.float32([x, y, z]))
        #newData = self.rotate(newData)
        self.data = self.data + newData
    
    def updateOccupancyMap(self):
        if self.robot.scene.occupancyMapType == self.robot.scene.OCCUPANCY_MAP_BINARY:
             self.occupancyMap = np.ones((self.hPix, self.wPix), np.uint8) * 255
             #r = int(self.l/2*self.m2pix()) # radius, option 1
             pointCloudPix = self.m2pix(self.dataCropped) # option 1
             for i in range(len(pointCloudPix)):
                 self.occupancyMap[(pointCloudPix[i][0], pointCloudPix[i][1])] = 0 # option 1
        
    def updateScanVector(self):
        self.scanVector = np.ones((1, self.lenScanVector), np.float32) * self.maxRange
        for i in range(len(self.data)):
            x = self.data[i][0]
            y = self.data[i][1]
            dist = (x ** 2 + y ** 2 ) ** 0.5
            k = math.ceil((math.atan2(y, x) + math.pi) 
                            / 2 / math.pi * self.lenScanVector) - 1 # bin index
            if self.scanVector[0, k] > dist:
                self.scanVector[0, k] = dist
            #print('dist: ', dist)
            
    def getObservation(self):
        if self.robot.scene.occupancyMapType == self.robot.scene.OCCUPANCY_MAP_BINARY:
            osbervation = self.occupancyMap.reshape((1, self.wPix * self.wPix))
        elif self.robot.scene.occupancyMapType == self.robot.scene.OCCUPANCY_MAP_THREE_CHANNEL:
            osbervation = self.occupancyMap.reshape((1, self.wPix * self.wPix * 3))
        return osbervation
    
    def rotate(self, data = None):
        if data is None:
            raise Exception('input cannot be None')
        alpha = self.robot.xi.alpha
        beta = self.robot.xi.beta
        gamma = self.robot.xi.theta
        Rx = self.getRotationMatrix([1, 0, 0], alpha)
        Ry = self.getRotationMatrix([0, 1, 0], beta)
        Rz = self.getRotationMatrix([0, 0, 1], gamma)
        R = Rx.dot(Ry)
        R = np.linalg.inv(R)
        #v = np.dot(R, np.dot(Rx, np.dot(Ry, np.dot(Rz, np.array([0, 0, 1])))))# for test
        #print("v = ", v)
        for i in range(len(data)):
            data[i] = np.dot(R, data[i])
        return data
        #self.show()
    
    def crop(self):
        self.dataCropped = []
        for i in range(len(self.data)):
            x = float(self.data[i][0])
            y = float(self.data[i][1])
            z = float(self.data[i][2])
            MIN = 0.20
            if any([x > self.xMax, x < -self.xMax, y > self.yMax, y < -self.yMax, z < -0.3]): #
                continue
            elif (x < MIN and y < MIN and x > -MIN and y > -MIN):
                continue
            self.dataCropped.append(np.float32([x, y]))
        #print('dataCropped length:', len(self.dataCropped))
    def getRotationMatrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis/math.sqrt(np.dot(axis, axis))
        a = math.cos(theta/2.0)
        b, c, d = -axis*math.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
            
    def m2pix(self, p = None):
        if p is None: # if p is None
            return (self.wPix / self.xMax / 2)
        xyPix = []
        for i in range(len(p)):
            xPix = ((self.xMax - p[i][0]) * (self.wPix / self.xMax / 2))
            yPix = ((self.yMax - p[i][1]) * (self.hPix / self.yMax / 2))
            xyPix.append(np.uint16([xPix, yPix]))
        return xyPix
    
    def show(self):
        pass
    
    
    
    
    
    