#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 13:12:51 2018

@author: cobus
"""

import numpy as np

class Kalman:
    
    ##KALMAN FILTER##
    
    def __init__(self, A, B, C, R, Q):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        
    
    def kalman_filter(self, meanPrev, sigPrev, control_t, z_t):
        A = self.A
        B = self.B
        C = self.C
        
        R = self.R
        Q = self.Q
        
        mean_vec_t = A.dot(meanPrev) + B.dot(control_t)
        sig_vec_t = A.dot(sigPrev).dot(A.T) + R
        
        K = sig_vec_t.dot(C.T).dot(np.linalg.inv(C.dot(sig_vec_t).dot(C.T) + Q))
        
        mean_t = mean_vec_t + K.dot(z_t - C.dot(mean_vec_t))
        
        sig_t = (np.identity(len(K)) - K.dot(C)).dot(sig_vec_t)
        
        return mean_t, sig_t 