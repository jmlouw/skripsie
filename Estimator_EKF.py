#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:10:46 2018

@author: cobus
"""

import numpy as np

class EKF:
    
    ##EKF##
    
    def __init__(self, R, Q):
        
        self.R = R
        self.Q = Q
        
    
    def func_EKF(self, meanPrev, sigPrev, control_t, z_t): #EKF algorithm... solves the non-linear localisation problem
        dt = 1
        theta_prev = meanPrev[2]
        
        vt = control_t[0]
        wt = control_t[1]
        
        Gt = np.array([[1,0, -(vt/wt)*np.cos(theta_prev) + (vt/wt)*np.cos(theta_prev + wt*dt)],
                       [0,1, -(vt/wt)*np.sin(theta_prev) + (vt/wt)*np.sin(theta_prev + wt*dt)],
                       [0,0, 1]])
    
        Ht = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])
    
        mean_stripe_t = meanPrev + np.array([-(vt/wt)*np.sin(theta_prev) + (vt/wt)*np.sin(theta_prev + wt*dt),
                                          (vt/wt)*np.cos(theta_prev) - (vt/wt)*np.cos(theta_prev + wt*dt),
                                          wt*dt])
    
        #print("\nmean_predict\n", mean_stripe_t)
       # print("mean_vec_t", mean_vec_t)
        sig_stripe_t = Gt.dot(sigPrev).dot(Gt.T) + self.R
        
        Kt  = sig_stripe_t.dot(Ht.T).dot(np.linalg.inv(Ht.dot(sig_stripe_t).dot(Ht.T) + self.Q))
        
        mean_t = mean_stripe_t + Kt.dot(z_t - Ht.dot(mean_stripe_t))
        
        sig_t = (np.identity(len(Kt)) - Kt.dot(Ht)).dot(sig_stripe_t)
    
        return mean_t, sig_t 