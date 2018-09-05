#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 22:12:03 2018

@author: cobus
"""

import numpy as np
from gauss import Gaussian

import copy
class pgm_estimator:
    
    def __init__(self, A, B, C, R, Q, measurements, controls, init_belief):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        
        self.controls = controls
        self.measurements = measurements
        
        self.messages_right = []
        self.messages_left = []
        self.beliefs = []
        
        self.init_belief = init_belief
        self.beliefs.append(init_belief)
        
        self.estimate_beliefs()
        self.beliefs = self.beliefs + self.messages_right
        
                     
    def upward_message(self, z_t):
        Q_Inv = np.linalg.inv(self.Q)
        myMatrix = np.concatenate((np.identity(len(self.C)), -self.C.T))
        k_ac = myMatrix.dot(Q_Inv).dot(myMatrix.T)
        h_ac = np.zeros(len(k_ac))        
        fac_zt_xt = Gaussian(RVs = ["zt1", "zt2", "xt1", "xt2"],K = k_ac, h = h_ac)
        fac_zt_xt.evidenceUpdate(np.array([0,1]), z_t)
        
        return fac_zt_xt
    
    def right_going_messages(self):
        R_Inv = np.linalg.inv(self.R)
        myMatrix = np.concatenate((np.identity(len(self.A)), -self.A.T, -self.B.T))       
        k_ac = myMatrix.dot(R_Inv).dot(myMatrix.T)       
        h_ac = np.zeros(len(k_ac))            
        fac_xt_ut_xtprev = Gaussian(RVs = ["xt1", "xt2", "xtPrev1", "xtPrev2",  "ut1" ,  "ut2"],K = k_ac, h = h_ac)
        
        fac_xt_ut_xtprev.evidenceUpdate(np.array([4,5]), self.controls[:,0])
        
        myBeliefXtPrev = copy.deepcopy(self.init_belief)
        myBeliefXtPrev.extendScope("xt1")
        myBeliefXtPrev.extendScope("xt2")
       
        myBeliefXtPrev.reArrangeEntries(np.array([2,3,0,1]))
        beliefXt_ = fac_xt_ut_xtprev.multiplyNew(myBeliefXtPrev)
        beliefXt_.marginalizeIndicesUpdate(np.array([2,3]))
 
        
        message_upward_initial = self.upward_message(self.measurements[:,1])        
        message_right_initial = beliefXt_.multiplyNew(message_upward_initial)
        
        self.messages_right.append(message_right_initial)
        #
        
        for i in range(1, len(self.controls[0])):##nie seker oor die -1
            #self.messages_right[i-1].toStringCan()
            fac_xt_ut_xtprev = Gaussian(RVs = ["xt1", "xt2", "xtPrev1", "xtPrev2",  "ut1" ,  "ut2"],K = k_ac, h = h_ac)
            
            message_upward = self.upward_message(self.measurements[:,i+1])
           # message_upward.toStringCan()
            message_upward.extendScope("xtPrev1")
            message_upward.extendScope("xtPrev2")
            
            fac_xt_ut_xtprev.evidenceUpdate(np.array([4,5]), self.controls[:,i])
            
            fac_xt_ut_xtprev.multiplyUpdate(message_upward)
            
            message_right_incoming = copy.deepcopy(self.messages_right[i-1])
            message_right_incoming.extendScope("xt1")
            message_right_incoming.extendScope("xt2")
            message_right_incoming.reArrangeEntries(np.array([2,3,0,1]))
            
            fac_xt_ut_xtprev.multiplyUpdate(message_right_incoming)
            #fac_xt_ut_xtprev.toStringCan()
            fac_xt_ut_xtprev.marginalizeIndicesUpdate(np.array([2,3])) 
            #fac_xt_ut_xtprev.toStringCan()
            self.messages_right.append(fac_xt_ut_xtprev)
            
        for i in range(0, len(self.controls[0])):
            self.messages_right[i].updateCov()
            #self.messages_right[i].toStringCov()
            
    def left_going_messages(self):
        pass
    
    def estimate_beliefs(self):
        self.right_going_messages()
        #self.left_going_messages()
        
#        R_Inv = np.linalg.inv(self.R)
#        myMatrix = np.concatenate((np.identity(len(self.A)), -self.A.T, -self.B.T))       
#        k_ac = myMatrix.dot(R_Inv).dot(myMatrix.T)       
#        h_ac = np.zeros(len(k_ac)) 
#                
#        for i in range (0, len(self.controls)):
#            fac_xt_ut_xtprev = Gaussian(RVs = ["xt1", "xt2", "xtPrev1", "xtPrev2",  "ut1" ,  "ut2"],K = k_ac, h = h_ac)
#            fac_xt_ut_xtprev.evidenceUpdate(np.array([4,5]), self.controls[i])
        
    
