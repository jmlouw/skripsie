#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 21:02:58 2018

@author: cobus
"""
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt


x = np.array([1,2,3,4,5,6,], dtype = np.float64)
y = np.array([5,4,6,5,6,7], dtype = np.float64)



m_mean = np.array([0,0,0])

cov = np.array([[1,0,0],
                [0,1,0],
                [0,0,1]])
xs = []
ys = []

v= 2
w =0.2
dt =1
for i in range (1000):
    xs.append(np.random.multivariate_normal(m_mean, cov))
    theta_prev = xs[i][2]
    ys.append(xs[i] + np.array([-(v/w)*np.sin(theta_prev) + (v/w)*np.sin(theta_prev + w*dt),
                                          (v/w)*np.cos(theta_prev) - (v/w)*np.cos(theta_prev + w*dt),
                                          w*dt]))
        
    
xs_arr = np.asarray(xs, dtype = np.float64).T
ys_arr = np.asarray(ys, dtype = np.float64).T




def best_fit_slope_and_intercept(x_s, y_s):
    print(x_s.shape)
    print(y_s.shape)
    m = ( ( (mean(x_s)* mean(y_s)) - mean(x_s*y_s) ) /
         ((mean(x_s)*mean(x_s)) - mean(x_s*x_s)) )
    
    b = mean(y_s) - m*mean(x_s)
    
    return m, b

 
mx, bx = best_fit_slope_and_intercept(xs_arr[0], ys_arr[0])
my, by = best_fit_slope_and_intercept(xs_arr[1], ys_arr[1])
mtheta, btheta = best_fit_slope_and_intercept(xs_arr[2], ys_arr[2])

#Fa = np.array([mx, 0, 0],
#              [0, my, 0],
#              [0, 0, mtheta])
#ga = np.array([bx, by, btheta])

line1 = [(mx*x) + bx for x in xs_arr[0]]
line2 = [(my*x) + by for x in xs_arr[1]]
line3 = [(mtheta*x) + btheta for x in xs_arr[2]]
 

plt.scatter(xs_arr[0], ys_arr[0])
plt.plot(xs_arr[0], line1)
plt.show()

plt.scatter(xs_arr[1], ys_arr[1])
plt.plot(xs_arr[1], line2)
plt.show()

plt.scatter(xs_arr[2], ys_arr[2])
plt.plot(xs_arr[2], line3)
plt.show()