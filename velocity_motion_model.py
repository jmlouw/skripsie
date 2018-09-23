#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 10:55:40 2018

@author: cobus
"""

import numpy as np

class vmm:
    @staticmethod
    def sim_move(controls, dt, R, Q):
        dt = 1
        xt = np.array(np.zeros((3,len(controls[0])+1)))
        zt = np.array(np.zeros((3,len(controls[0])+1)))
        
        mov_noise_mean = np.array([0,0,0])
        mov_noise = np.random.multivariate_normal(mov_noise_mean, R)        
        
        mes_noise_mean = np.array([0,0,0])
        mes_noise = np.random.multivariate_normal(mes_noise_mean, Q)        
        
        xt[:,0] = xt[:,0] + mov_noise
        zt[:,0] = xt[:,0] + mes_noise

        for i in range(0, len(controls[0])):  
            x_prev = xt[0, i]
            y_prev = xt[1, i]
            theta_prev = xt[2, i]
        
            v = controls[0,i]
            w = controls[1,i]
   
            x_t = x_prev - (v/w)*np.sin(theta_prev) + (v/w)*np.sin(theta_prev + w*dt)
            y_t = y_prev + (v/w)*np.cos(theta_prev) - (v/w)*np.cos(theta_prev + w*dt)
            theta_t = theta_prev + w*dt
    
            mov_noise = np.random.multivariate_normal(mov_noise_mean, R)        
            xt[:,i+1] = np.array([x_t,y_t,theta_t])+mov_noise
    
            mes_noise = np.random.multivariate_normal(mes_noise_mean, Q)
            zt[:,i+1] = xt[:,i+1] + mes_noise        
        return xt, zt