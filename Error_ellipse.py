#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 08:48:58 2018

@author: cobus
"""
#import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

class e_ellipse:
    @staticmethod
    def plot_eEllipse(mean, sigma, ax , col):        
           # ax1 = plt.subplot(111, aspect = 'equal')
            eigValues, eigVectors= np.linalg.eig(sigma[0:2,0:2])            
            maxIndex = np.argmax(eigValues)
            minIndex = 1 - maxIndex
#            print("\neigVectors\n", eigVectors)
#            print("\neigVaules\n",eigValues)
#            print(np.argmax(eigValues))            
            alpha = 0
            if (eigVectors[1, maxIndex] != 0):
                alpha = np.arctan(eigVectors[0,maxIndex]/eigVectors[1, maxIndex])
            #print("\nalpha\n", alpha)            
            #estimatedLambdas[:, i] = np.sqrt(estimatedLambdas[:, i])                
            ell = Ellipse(xy = (mean[0],mean[1]), 
                        width = 2*np.sqrt(eigValues[maxIndex]*5.991), 
                        height = 2*np.sqrt(eigValues[minIndex]*5.991), 
                        angle = np.rad2deg(alpha), color = col)            
            ell.set_facecolor('none')
            ax.add_artist(ell)