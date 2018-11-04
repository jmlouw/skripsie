#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:22:40 2018

@author: cobus
"""
import numpy as np
from gauss import Gaussian
from statistics import mean
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc
import copy
import time

class BayesNonLinear:
    
    ##KALMAN FILTER##
    
    def __init__(self, R, Q, N_samples = None):
        self.R = R
        self.Q = Q
        self.N_samples = N_samples
        
    
    def estimate(self, beliefXtPrev, control_t, z_t, mode):#solves the linear localisation problem with the more general bayes filter        
        myBeliefXtPrev = copy.deepcopy(beliefXtPrev) #deepcopy?
        
        start = time.time()
        if(mode == 0):
            Fa, ga, sig_n = self.linearize_taylor(beliefXtPrev, control_t)
        if(mode == 1):
            Fa, ga, sig_n = self.linearize_unscented(beliefXtPrev, control_t)
        if(mode == 2):
            Fa, ga, sig_n = self.lob(beliefXtPrev, control_t)
        if(mode == 3):
            Fa, ga, sig_n = self.random_estimate(beliefXtPrev, control_t)
        end = time.time() 
        exTime = end - start
            
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
        
        return beliefXt, exTime #returns the belief of the Xt
    
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
        
        return Fa, ga, self.R #self.R reg hier?
    
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
        
#        print("sig xy shape", sig_xy.shape)
        
        sig_xx_inv = np.linalg.inv(sig_xx)        
        Fa = sig_xy.T.dot(sig_xx_inv)        
        ga =mean_y - Fa.dot(mean_x)
        sig_n = sig_yy - sig_xy.T.dot(sig_xx_inv).dot(sig_xy)  
        
        return Fa, ga, sig_n
    
    def random_estimate(self, beliefXtPrev, control_t):
        xs_arr, ys_arr = self.random_data(beliefXtPrev, control_t)
        #print(ys_arr.shape)
        mean_x = beliefXtPrev.mean
        sig_xx = beliefXtPrev.cov  
        
        mean_y = np.mean(ys_arr, axis = 1)
        sig_yy = np.cov(ys_arr)
        
        sig_yy += self.R
        
#        print("\nMean\n", mean_y)
#        print("\nSig_yy\n", sig_yy)
        
        sig_xy = np.array(np.zeros((3,3)))
        for i in range(len(xs_arr[0])):
            t1 = xs_arr[:,i] - mean_x
            t2 = ys_arr[:,i] - mean_y
            #z_hat==mean_predict     
            sig_xy += np.outer(t1,t2)        
        sig_xy = sig_xy/len(xs_arr[0])
        
#        mean_xy = np.mean(xs_arr.dot(ys_arr.T), axis = 0)
#        print("\nmeanxy\n", mean_xy)
#        sig_xy = mean_xy - np.outer(mean_x, mean_y)
        
        
#        print("\nsig_xy\n", sig_xy)
#        
        sig_xx_inv = np.linalg.inv(sig_xx)        
        Fa = sig_xy.T.dot(sig_xx_inv)        
        ga =mean_y - Fa.dot(mean_x)
        sig_n = sig_yy - sig_xy.T.dot(sig_xx_inv).dot(sig_xy)  
        
        return Fa, ga, sig_n
        
    def random_data(self, beliefXtPrev, control_t):
        xs = []
        ys = []        
        mean_x = beliefXtPrev.mean
        sig_xx = beliefXtPrev.cov        
        v = control_t[0]
        w = control_t[1]
        dt =1
        
        for i in range (self.N_samples):
            xs.append(np.random.multivariate_normal(mean_x, sig_xx))
            theta_prev = xs[i][2]
            ys.append(xs[i] + np.array([-(v/w)*np.sin(theta_prev) + (v/w)*np.sin(theta_prev + w*dt),
                                          (v/w)*np.cos(theta_prev) - (v/w)*np.cos(theta_prev + w*dt),
                                          w*dt]))       
        xs_arr = np.asarray(xs, dtype = np.float64).T
        ys_arr = np.asarray(ys, dtype = np.float64).T        
        return xs_arr, ys_arr

    def lob(self, beliefXtPrev, control_t):
        xs_arr, ys_arr = self.random_data(beliefXtPrev, control_t)
        
        mxx, bxx = self.best_fit_slope_and_intercept(xs_arr[0], ys_arr[0])
#        mxy, bxy = self.best_fit_slope_and_intercept(xs_arr[0], ys_arr[1])
#        mxt, bxt = self.best_fit_slope_and_intercept(xs_arr[0], ys_arr[2])        
        myy, byy = self.best_fit_slope_and_intercept(xs_arr[1], ys_arr[1])
#        myx, byx = self.best_fit_slope_and_intercept(xs_arr[1], ys_arr[0])
#        myt, byt = self.best_fit_slope_and_intercept(xs_arr[1], ys_arr[2])        
        mtt, btt = self.best_fit_slope_and_intercept(xs_arr[2], ys_arr[2])
#        mtx, btx = self.best_fit_slope_and_intercept(xs_arr[2], ys_arr[0])
#        mty, bty = self.best_fit_slope_and_intercept(xs_arr[2], ys_arr[1])
        
#        Fa = np.array([[mxx, mxy, mxt],
#                      [myx, myy, myt],
#                      [mtx, mty, mtt]])
        
        Fa = np.array([[mxx, 0, 0],
                      [0, myy, 0],
                      [0, 0, mtt]])
        ga = np.array([bxx, byy, btt])
        
#        line1 = [(mxx*x) + bxx for x in xs_arr[0]]
#        line2 = [(myy*x) + byy for x in xs_arr[1]]
#        line3 = [(mtt*x) + btt for x in xs_arr[2]]
 
#        plt.scatter(xs_arr[0], ys_arr[0])
#        plt.plot(xs_arr[0], line1)
#        plt.show()
#
#        plt.scatter(xs_arr[1], ys_arr[1])
#        plt.plot(xs_arr[1], line2)
#        plt.show()
#
#        plt.scatter(xs_arr[2], ys_arr[2])
#        plt.plot(xs_arr[2], line3)
#        plt.show()
#        
        return Fa, ga, self.R
    

    def best_fit_slope_and_intercept(self, x_s, y_s):
       # print(x_s.shape)
       # print(y_s.shape)
        m = ( ( (mean(x_s)* mean(y_s)) - mean(x_s*y_s) ) /
             ((mean(x_s)*mean(x_s)) - mean(x_s*x_s)) )
    
        b = mean(y_s) - m*mean(x_s)
    
        return m, b
    
 
        
        