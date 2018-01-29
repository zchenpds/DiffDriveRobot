# DiffDriveRobot

## Install OpenCV as a dependency
### Spyder 3.6 on Windows
To install opencv, run Anaconda Prompt as administrator. Then execute the following.
```
conda install -c conda-forge opencv
```

## Get your Python program ready to be a Vrep client (Remote API)
To use the remote API functionality in your Python script, you will need following 3 items:
```
vrep.py
vrepConst.py
remoteApi.dll, remoteApi.dylib or remoteApi.so (depending on your target platform)
```
Above files are located in V-REP's installation directory, under programming/remoteApiBindings/python. You might have to build the remoteApi shared library yourself (using remoteApiSharedLib.vcproj or makefile) if not already built. In that case, make sure you have defined NON_MATLAB_PARSING and MAX_EXT_API_CONNECTIONS=255 as a preprocessor definition.

Once you have above elements in a directory known to Python, call import vrep to load the library. To enable the remote API on the client side (i.e. your application), call vrep.simxStart. See the simpleTest.py script in the programming/remoteApiBindings/python directory for an example. This page lists and describes all supported Python remote API functions. V-REP remote API functions can easily be recognized from their "simx"-prefix.

### Spyder 3.6 on Windows
In Spyder, go to menu Tools -> PYTHONPATH Manager. In there, add the paths "...\programming\remoteApiBindings\python\python" and "...\programming\remoteApiBindings\lib\lib\64Bit", where "..." is the installation location of Vrep.
