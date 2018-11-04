#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 10:55:40 2018

@author: cobus
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from gauss import Gaussian
from Error_ellipse import e_ellipse
from Estimator_Bayes_NonLinear import BayesNonLinear

class vmm:
    @staticmethod
    def sim_move(controls, dt, R, Q):
        dt = 1
        xt = []
        zt = []
        
        mov_noise_mean = np.array([0,0,0])
        mov_noise = np.random.multivariate_normal(mov_noise_mean, R)        
        
        mes_noise_mean = np.array([0,0,0])
        mes_noise = np.random.multivariate_normal(mes_noise_mean, Q)        
        
        xt.append(np.array([0,0,0]) + mov_noise)
        zt.append(xt[0] + mes_noise)

        for i in range(0, len(controls)):  
            x_prev = xt[i][0]
            y_prev = xt[i][1]
            theta_prev = xt[i][2]
        
            v = controls[i][0]
            w = controls[i][1]
            
   
            x_t = x_prev - (v/w)*np.sin(theta_prev) + (v/w)*np.sin(theta_prev + w*dt)
            y_t = y_prev + (v/w)*np.cos(theta_prev) - (v/w)*np.cos(theta_prev + w*dt)
            theta_t = theta_prev + w*dt
    
            mov_noise = np.random.multivariate_normal(mov_noise_mean, R)        
            xt.append(np.array([x_t,y_t,theta_t])+mov_noise)
    
            mes_noise = np.random.multivariate_normal(mes_noise_mean, Q)
            zt.append(xt[i+1] + mes_noise)        
        return xt, zt
    
    @staticmethod
    def testKL():
        dt = 1
        controls = []
       
        wt = (2*np.pi)/10
        R = np.array([[0.05,0,0],[0,0.05,0],[0,0,0.02]])
        Q = np.array([[0.05,0,0],[0,0.05,0],[0,0,0.02]])


        control = np.array([0,0], float)
        control[0] = np.random.normal(1.5, 0.5)
        control[1] = np.random.normal(wt,0.5)
        controls.append(control)
        
        xt, zt = vmm.sim_move(controls, dt, R, Q)
        
        mean_init = np.array([0,0,0])
        cov_init = np.array([[0.05, 0, 0],[0, 0.05, 0], [0, 0, 0.05]])
        
        
        belief_init = Gaussian(['x','y','theta'], mean = mean_init, cov = cov_init)
      #  print("init mean",belief_init.mean)
       
        bayesFilter = BayesNonLinear(R, Q, N_samples = 10000)
        groundTruth , gtTime = bayesFilter.estimate(belief_init, controls[0], zt[1], mode = 3)
        
        bayesFilter = BayesNonLinear(R, Q, N_samples = 100)
        monte100 , monte100Time = bayesFilter.estimate(belief_init, controls[0], zt[1], mode = 3)
        
        bayesFilter = BayesNonLinear(R, Q, N_samples = 10)
        monte10, monte10Time = bayesFilter.estimate(belief_init, controls[0], zt[1], mode = 3)
        
        unscented, ukftime = bayesFilter.estimate(belief_init, controls[0], zt[1], mode = 3)
        
        taylor, talortime = bayesFilter.estimate(belief_init, controls[0], zt[1], mode = 3)     
        
            
       
        
        kl_gt = vmm.KL(groundTruth, groundTruth)
        kl_unscent = vmm.KL(groundTruth, unscented)
        kl_100 = vmm.KL(groundTruth, monte100)
        kl_10 = vmm.KL(groundTruth, monte10)        
        kl_taylor = vmm.KL(groundTruth, taylor)
        
#        print("ground truth", kl_gt)
#        print("monte100", kl_100)
#        print("monte10", kl_10)
#        print("unscented", kl_unscent)
#        print("taylor", kl_taylor)
#        plt.figure(0)
#        plt.subplot(111,  aspect = 'equal')
#        vmm.myplot(belief_init, "black")
#        vmm.myplot(groundTruth, "red")
#        vmm.myplot(monte100, "orange")
#        vmm.myplot(monte10, "brown")
#        vmm.myplot(unscented, "blue")
#        vmm.myplot(taylor, "green")
#        
#        
#        custom_lines = [Line2D([0], [0], color= "black", lw=4),
#                Line2D([0], [0], color="red", lw=4),
#                Line2D([0], [0], color="orange", lw=4),
#                Line2D([0], [0], color="brown", lw=4),
#                Line2D([0], [0], color="blue", lw=4),
#                Line2D([0], [0], color="green", lw=4)]
#
#        plt.legend(custom_lines, ('Common prior belief', 'Ground truth', 'Monte Carlo 100 samples', 'Monte Carlo 10 samples', 'Unscented transform', 'Taylor expansion'))
        #return(np.array([gtTime, monte100Time,monte10Time, ukftime,talortime]))
        return(np.array([kl_unscent,kl_100,kl_10,kl_taylor]))
        
        
  
       
        
        
    def myplot(mybelief, myColor):
        mybelief.marginalizeIndicesUpdate(arrIndices = np.array([2]))
        mybelief.updateCov()
        
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        e_ellipse.plot_eEllipse(mybelief.mean, mybelief.cov, col = myColor)
        plt.scatter(mybelief.mean[0], mybelief.mean[1], c = myColor, marker = ".")
        
    def KL(dist1, dist2):
        dist1.updateCov()
        dist2.updateCov()
        sigma1 = dist1.cov
        sigma2 = dist2.cov
        mean1 = dist1.mean
        mean2 = dist2.mean
        n = len(mean1)
       # print(n)
        part1 = np.log((np.linalg.det(sigma2))/(np.linalg.det(sigma1)))
        part2 = np.trace(np.linalg.inv(sigma2).dot(sigma1))
       # print(part2)
        part3 = (np.transpose(mean2-mean1)).dot(np.linalg.inv(sigma2)).dot(mean2-mean1)
      
        answer = 0.5*(part1 - n + part2 + part3)
        return answer
        
        
        
        
   
        