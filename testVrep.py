# Copyright 2006-2017 Coppelia Robotics GmbH. All rights reserved. 
# marc@coppeliarobotics.com
# www.coppeliarobotics.com
# 
# -------------------------------------------------------------------
# THIS FILE IS DISTRIBUTED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
# WARRANTY. THE USER WILL USE IT AT HIS/HER OWN RISK. THE ORIGINAL
# AUTHORS AND COPPELIA ROBOTICS GMBH WILL NOT BE LIABLE FOR DATA LOSS,
# DAMAGES, LOSS OF PROFITS OR ANY OTHER KIND OF LOSS WHILE USING OR
# MISUSING THIS SOFTWARE.
# 
# You are free to use/modify/distribute this file for whatever purpose!
# -------------------------------------------------------------------
#
# This file was automatically created for V-REP release V3.4.0 rev. 1 on April 5th 2017

# Make sure to have the server side running in V-REP: 
# in a child script of a V-REP scene, add following command
# to be executed just once, at simulation start:
#
# simExtRemoteApiStart(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

import sys

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time

print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
objs = []
if clientID!=-1:
    print ('Connected to remote API server')
    
    # Start the simulation:
    vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    
    res, robotHandle = vrep.simxGetObjectHandle(clientID, 'Pioneer_p3dx', vrep.simx_opmode_oneshot_wait)
    res, motorLeft = vrep.simxGetObjectHandle(clientID, "Pioneer_p3dx_leftMotor", vrep.simx_opmode_oneshot_wait)
    res, motorRight = vrep.simxGetObjectHandle(clientID, "Pioneer_p3dx_rightMotor", vrep.simx_opmode_oneshot_wait)
    
    startTime=time.time()
    vrep.simxGetObjectPosition(clientID, robotHandle, -1, vrep.simx_opmode_streaming)
    while time.time()-startTime < 5:
        res, pos = vrep.simxGetObjectPosition(clientID, robotHandle, -1, vrep.simx_opmode_buffer)
        if res == vrep.simx_return_ok: 
            #print ('Position: ', pos)
            print('Time: ', time.time() - startTime)
        vrep.simxSetJointTargetVelocity(clientID, motorLeft, -0.5, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(clientID, motorRight, 0.5, vrep.simx_opmode_oneshot)
        time.sleep(0.005)
        

    # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    vrep.simxGetPingTime(clientID)
    
    
    # Stop simulation:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    
    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')

print ('Program ended')
