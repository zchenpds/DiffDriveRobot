# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:49:53 2018

@author: cz
"""

from sceneplot import ScenePlot
import saver

sc = saver.load(4)
sp = ScenePlot(sc)
sp.plot(sp.TYPE_TIME_ACTIONS)