# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 16:39:54 2016

@author: YI
"""

import numpy as np

#read data
##data = pd.read_csv('ex1data2.txt', header=None)
data = np.loadtxt('ex1data2.txt', delimiter=',')
x = data[:,0:2]
y = data[:,2]
m = len(y)
y = y.reshape((m,1))

#normalize the features
def normalize(x):
    n = len(x[0])
    for i in xrange(n):
        mean = np.mean(x[:,i])
        std = np.std(x[:,i])
        x[:,i] = (x[:,i]-mean)/std
    return x
    
x = normalize(x)

#add intercept term to X
X = np.ones((m,3))
X[:,1:] = x   #47*3

#Loss function
##initialize the parameters
theta = np.zeros((3,1))   #3*1
num_iters = 20000
alpha = 0.01

def loss(x,y,theta):
    L = np.sum((x.dot(theta)-y)**2)/(2*m)
    return L

loss(X,y,theta)    

#create a gradient descent function
def gradient_descent(x, y, theta, alpha, num_iters):
    
    p = len(x[0])    
    L_history = np.zeros((num_iters, 1))

    for i in xrange(num_iters):
        
        for f in xrange(p):
            
            dtheta = np.sum((x.dot(theta)-y)*x[:,f])/m
            
            theta[f][0] = theta[f][0] - alpha * dtheta
            
    L_history[i, 0] = loss(x, y, theta)

    return theta, L_history
    
gradient_descent(X,y,theta,alpha,num_iters)
 
a=np.array((1,1650,3))
predict = a.dot(theta)


