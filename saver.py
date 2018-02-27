# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:33:11 2018

@author: cz
"""

# save scene
import pickle
import os

directory = 'data_scene'

def save(sc):
    if not os.path.exists(directory):
        os.makedirs(directory)
    count = 0
    for filename in os.listdir(directory):
        name, _ = os.path.splitext(filename)
        n = int(name[3:])
        if count < n:
            count = n
    count += 1
    for robot in sc.robots:
        if robot.learnedController is not None:
            robot.learnedController = None
    with open(os.path.join(directory, 'sc'+str(count).zfill(3)+'.pkl'), 'wb') as f:
        pickle.dump(sc, f)

def load(count):
    with open(os.path.join(directory, 'sc'+str(count).zfill(3)+'.pkl'), 'rb') as f:
        sc = pickle.load(f)
        return sc