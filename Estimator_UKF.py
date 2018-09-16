#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:02:37 2018

@author: cobus
"""

import numpy as np

class UKF:    
    
    def __init__(self, R, Q):
        
        self.R = R
        self.Q = Q
        
    
    def func_UKF(self, meanPrev, sigPrev, control_t, z_t): #EKF algorithm... solves the non-linear localisation problem
        ##lambda + n = 1
        gamma = 1
        dt = 1
        theta_prev = meanPrev[2]
        mean_t = 0
        sig_t = 0
        n = len(meanPrev)
        
        temp1 = meanPrev + gamma*np.linalg.cholesky(sigPrev)
        temp2 = meanPrev - gamma*np.linalg.cholesky(sigPrev)
        
        print(np.expand_dims(meanPrev, axis=1))
        print(temp1)
        print(temp2)
        
        sigmaPointsPrev = np.concatenate((np.expand_dims(meanPrev, axis=1), temp1, temp2 ), axis = 1)
        
        sigPointStar = []
        
        print(len(sigmaPointsPrev[0]))
        
        vt = control_t[0]
        wt = control_t[1]
        
        wm = []
        wc = []
        
        for i in range(len(sigmaPointsPrev[0])):
            theta_prev = sigmaPointsPrev[2,i]
            sigPointStar.append(sigmaPointsPrev[:,i] + np.array([-(vt/wt)*np.sin(theta_prev) + (vt/wt)*np.sin(theta_prev + wt*dt),
                                          (vt/wt)*np.cos(theta_prev) - (vt/wt)*np.cos(theta_prev + wt*dt),
                                          wt*dt]))
        
        
    
        print(len(sigPointStar))
        print(sigPointStar[0])
    
        return mean_t, sig_t 