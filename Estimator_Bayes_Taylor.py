#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:22:40 2018

@author: cobus
"""
import numpy as np
from gauss import Gaussian

import copy
class BayesTaylor:
    
    ##KALMAN FILTER##
    
    def __init__(self, R, Q):
        self.R = R
        self.Q = Q
        
    
    def estimate(self, beliefXtPrev, control_t, z_t, mode):#solves the linear localisation problem with the more general bayes filter        
        myBeliefXtPrev = copy.deepcopy(beliefXtPrev) #deepcopy?
        
        if(mode == 0):
            Fa, ga, sig_n = self.linearize_taylor(beliefXtPrev, control_t)
        if(mode == 1):
            Fa, ga, sig_n = self.linearize_unscented(beliefXtPrev, control_t)
            
        sig_n_inv = np.linalg.inv(sig_n)       
        tempMatrix = np.concatenate((-Fa.T,np.identity(len(Fa))))          
        k_ac = tempMatrix.dot(sig_n_inv).dot(tempMatrix.T)        
        h_ac = tempMatrix.dot(sig_n_inv).dot(ga)
        
        fac_xt_ut_xtprev = Gaussian(RVs = ["xPrev", "yPrev", "thetaPrev", "x", "y","theta"],K = k_ac, h = h_ac)        
  
        myBeliefXtPrev.extendScope("x")
        myBeliefXtPrev.extendScope("y")
        myBeliefXtPrev.extendScope("theta")
        
        beliefXt_ = fac_xt_ut_xtprev.multiplyNew(myBeliefXtPrev)
        beliefXt_.marginalizeIndicesUpdate(np.array([0,1,2]))
    
        H = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])    
        
        Q_Inv = np.linalg.inv(self.Q)
        myMatrix = np.concatenate((np.identity(len(H)), -H.T))
        k_ac = myMatrix.dot(Q_Inv).dot(myMatrix.T)
        h_ac = np.zeros(len(k_ac))        
        fac_zt_xt = Gaussian(RVs = ["z_x", "z_y", "z_theta", "x", "y", "theta"],K = k_ac, h = h_ac)
        fac_zt_xt.evidenceUpdate(np.array([0,1, 2]), z_t)
        
        beliefXt = fac_zt_xt.multiplyNew(beliefXt_)
        
        return beliefXt #returns the belief of the Xt
    
    def linearize_taylor(self, beliefXtPrev, control_t):       
        meanPrev = beliefXtPrev.mean
        dt = 1
        theta_prev = meanPrev[2]
        
        v = control_t[0]
        w = control_t[1]
        
        Fa = np.array([[1,0, -(v/w)*np.cos(theta_prev) + (v/w)*np.cos(theta_prev + w*dt)], ##F
                       [0,1, -(v/w)*np.sin(theta_prev) + (v/w)*np.sin(theta_prev + w*dt)],
                       [0,0, 1]])

        meanPredict = meanPrev + np.array([-(v/w)*np.sin(theta_prev) + (v/w)*np.sin(theta_prev + w*dt),
                                          (v/w)*np.cos(theta_prev) - (v/w)*np.cos(theta_prev + w*dt),
                                          w*dt])
        
        ga = meanPredict - Fa.dot(meanPrev)
        
        return Fa, ga, self.R
    
    def linearize_unscented(self, beliefXtPrev, control_t):
        dt = 1
        mean_x = beliefXtPrev.mean
        sig_xx = beliefXtPrev.cov
        n = len(mean_x)
        gamma = 1
        beta = 2
        kappa = 0
        lam = 1 - n
        alpha_square = (lam+n)/(n+kappa)
        m = np.expand_dims(mean_x, axis =1)
        temp1 = m + gamma*np.linalg.cholesky(sig_xx)
        temp2 = m - gamma*np.linalg.cholesky(sig_xx)        
        #line 2
        sigmaPointsPrev = np.concatenate((m, temp1, temp2), axis =1)        
        sigPointStar = []
        wm0 = (lam/(n+lam))    
        wc0 = (lam/(n+lam) + (1 - alpha_square + beta))                   
        wi = 1/(2*(n + lam))        
        vt = control_t[0]
        wt = control_t[1]
            
        #line 3
        for i in range(len(sigmaPointsPrev[0])):
            theta_prev = sigmaPointsPrev[2,i]
            tempVar = np.array([-(vt/wt)*np.sin(theta_prev) + (vt/wt)*np.sin(theta_prev + wt*dt),
                                (vt/wt)*np.cos(theta_prev) - (vt/wt)*np.cos(theta_prev + wt*dt),
                                wt*dt])    
            result = sigmaPointsPrev[:,i] + tempVar
            sigPointStar.append(result)        
        
        #line4
        mean_y = np.array(np.zeros(n))
        for i in range(len(sigPointStar)):
            wm = wm0
            if (i != 0):
                wm = wi
            mean_y += wm*sigPointStar[i]
            
        
        #line5
        sig_yy = np.array(np.zeros((n,n)))
        for i in range(len(sigPointStar)):
            wc = wc0
            if (i != 0):
                wc = wi
            temp_c = sigPointStar[i]-mean_y
            sig_yy += wc* np.outer(temp_c, temp_c)#(temp_c.dot(np.transpose(temp_c)))
        sig_yy += self.R
        
        sig_xy = np.array(np.zeros((n,n)))
        for i in range(2*n+1):
            wc = wc0
            if (i != 0):
                wc = wi
            t1 = sigmaPointsPrev[:,i] - mean_x
            t2 = sigPointStar[i] - mean_y
            #z_hat==mean_predict     
            sig_xy += wc*np.outer(t1,t2)
        
        sig_xx_inv = np.linalg.inv(sig_xx)        
        Fa = sig_xy.T.dot(sig_xx_inv)        
        ga =mean_y - Fa.dot(mean_x)
        sig_n = sig_yy - sig_xy.T.dot(sig_xx_inv).dot(sig_xy)  
        
        return Fa, ga, sig_n
    
        