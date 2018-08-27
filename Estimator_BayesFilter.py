#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:22:40 2018

@author: cobus
"""
import numpy as np
from gauss import Gaussian

import copy
class Bayes:
    
    ##KALMAN FILTER##
    
    def __init__(self, A, B, C, R, Q):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        
    
    def bayes_filter(self, beliefXtPrev, control_t, z_t): #solves the linear localisation problem with the more general bayes filter
        myBeliefXtPrev = copy.deepcopy(beliefXtPrev) ##deepcopy?
        R_Inv = np.linalg.inv(self.R)
        myMatrix = np.concatenate((np.identity(len(self.A)), -self.A.T, -self.B.T))
       
        k_ac = myMatrix.dot(R_Inv).dot(myMatrix.T)
       
        h_ac = np.zeros(len(k_ac))    
        
        fac_xt_ut_xtprev = Gaussian(RVs = ["xt1", "xt2", "xtPrev1", "xtPrev2",  "ut1" ,  "ut2"],K = k_ac, h = h_ac)        
        
        fac_xt_ut_xtprev.evidenceUpdate(np.array([4,5]), control_t)
  
        myBeliefXtPrev.extendScope("xt1")
        myBeliefXtPrev.extendScope("xt2")
       
        myBeliefXtPrev.reArrangeEntries(np.array([2,3,0,1]))
 
        beliefXt_ = fac_xt_ut_xtprev.multiplyNew(myBeliefXtPrev)
        beliefXt_.marginalizeIndicesUpdate(np.array([2,3]))
        
        #beliefXt_.toStringCan()
        #beliefXt_.evidenceUpdate(np.array([4,5]), control_t)
        
        Q_Inv = np.linalg.inv(self.Q)
        myMatrix = np.concatenate((np.identity(len(self.C)), -self.C.T))
        k_ac = myMatrix.dot(Q_Inv).dot(myMatrix.T)
        h_ac = np.zeros(len(k_ac))        
        fac_zt_xt = Gaussian(RVs = ["zt1", "zt2", "xt1", "xt2"],K = k_ac, h = h_ac)
        fac_zt_xt.evidenceUpdate(np.array([0,1]), z_t)
        
        beliefXt = fac_zt_xt.multiplyNew(beliefXt_)
        
        return beliefXt #returns the belief of the Xt