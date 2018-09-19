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
    
    def func_UKF(self, meanPrev, sigPrev, control_t, z): #EKF algorithm... solves the non-linear localisation problem
        dt = 1
        theta_prev = meanPrev[2]
        
        n = len(meanPrev)
        #print("n", n)
        gamma = 1
        beta = 2
        kappa = 0
        lam = 1 - n
        alpha_square = (lam+n)/(n+kappa)
        #print("alpha_square", alpha_square)
        
        temp1 = meanPrev + gamma*np.linalg.cholesky(sigPrev)
        temp2 = meanPrev - gamma*np.linalg.cholesky(sigPrev)
        
#        print(np.expand_dims(meanPrev, axis=1))
#        print("\ntemp1\n", temp1)
#        print("\ntemp2\n", temp2)
        
        #line 2
        sigmaPointsPrev = np.concatenate((np.expand_dims(meanPrev, axis=1), temp1, temp2 ), axis = 1)
        
 #       print("\nsigmaPointsPrev\n", sigmaPointsPrev)
#                
        sigPointStar = []
       # print(len(sigmaPointsPrev[0]))       
     
        wm0 = (lam/(n+lam))    
        wc0 = (lam/(n+lam) + (1 - alpha_square + beta))                   
        wi = 1/(2*(n + lam))
        
#        print("wm0:", wm0)
#        print("wc0:", wc0)
#        print("wi:", wi)
        
        vt = control_t[0]
        wt = control_t[1]
            
        #line 3
        for i in range(len(sigmaPointsPrev[0])):
            theta_prev = sigmaPointsPrev[2,i]
            tempVar = np.array([-(vt/wt)*np.sin(theta_prev) + (vt/wt)*np.sin(theta_prev + wt*dt),
                                                                  (vt/wt)*np.cos(theta_prev) - (vt/wt)*np.cos(theta_prev + wt*dt),
                                                                  wt*dt])
            result = sigmaPointsPrev[:,i] + tempVar
#            print("\nsigmaPointsPrev\n", sigmaPointsPrev[:,i])
#            print("\ntempVar\n", tempVar)
#            print("\nResult\n", result)
            
            sigPointStar.append(result)
            
        
        #print("\nsigPointStar\n", sigPointStar)
        
        #line4
        mean_predict = np.array(np.zeros(n))
        for i in range(2*n+1):
            wm = wm0
            if (i != 0):
                wm = wi
#            print("\nmean_predict", mean_predict)
#            print("sigPointStar\n", wm*sigPointStar[i])
            mean_predict += wm*sigPointStar[i]
            
        
        #print("\nmean_predict\n", mean_predict)
        
        #line5
        sig_predict = np.array(np.zeros((n,n)))
        for i in range(2*n+1):
            wc = wc0
            if (i != 0):
                wc = wi
                
            temp_c = sigPointStar[i]-mean_predict
            sig_predict += wc*temp_c.dot(np.transpose(temp_c))
        sig_predict += self.R
        
        #line6
        temp3 = mean_predict + gamma*np.linalg.cholesky(sig_predict)
        temp4 = mean_predict - gamma*np.linalg.cholesky(sig_predict)
        sigPointPredict = np.concatenate((np.expand_dims(mean_predict, axis = 1), temp3, temp4), axis = 1)
        
       #print("\nsigPointPredict\n", sigPointPredict)
        #line7
        z_predict = sigPointPredict
        
        #line8
        z_hat = np.array(np.zeros(n))
        for i in range(2*n+1):
#            print(z_predict[:,i])
            wm = wm0
            if (i != 0):
                wm = wi
            z_hat += wm*z_predict[:,i]
            
        #line9
        S = np.array(np.zeros((n,n)))
        for i in range(2*n+1):
            wc = wc0
            if (i != 0):
                wc = wi
            temp_s = z_predict[:,i]-z_hat
            S += wc*temp_s.dot(np.transpose(temp_s))
        S += self.Q
        
        #line10
        sig_predict_xz = np.array(np.zeros((n,n)))
        for i in range(2*n+1):
            wc = wc0
            if (i != 0):
                wc = wi
            sig_predict_xz += wc*(sigPointPredict[:,i] - mean_predict).dot(np.transpose(z_predict[:,i] - z_hat))
        
        K = sig_predict_xz.dot(np.linalg.inv(S))
        
        mean = mean_predict + K.dot(z- z_hat)
        sig = sig_predict - K.dot(S).dot(np.transpose(K))
#        print(len(sigPointStar))
#        print(sigPointStar[0])       
    
        return mean, sig 