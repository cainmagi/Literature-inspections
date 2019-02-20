#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Project - Test Gibbs sampling algorithm and
#           its convergence
# Designed by Yuchen Jin (cainmagi)
# Requirements: 
#   python 3.6+, numpy 1.14+, scipy 1.1+, 
#   matplotlib 3+
# Version 1.0, Date: 2019/02/20
# Comments:
#   A numpy simulation of Gibbs sampling algorithm
#   In this code, generate the samples from a 2D
#   distribution by applying Gibbs sampling. And
#   we would also estimate the standard error.
####################################################
'''
import inspect
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import triang
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Traingle2D:
    '''
    2D traingle distribution
        only provide method to calculate pdf.
        need to use Metropolis-Hastings algorithm to get sample.
    '''
    def __init__(self, x_mu, y_mu, x_s, y_s):
        self.x_mu = x_mu
        self.y_mu = y_mu
        self.x_s = x_s
        self.y_s = y_s
        self.c_sum = (x_s * y_s * 2)/3
        self.fixpdf = multivariate_normal([x_mu, y_mu], np.diag([10*x_s, 10*y_s]))
        
    def __call__(self, x):
        x = np.array(x)
        if x.ndim == 2:
            return self.pdf_xlist(x)
        else:
            return self.pdf(x)
        
    def pdf(self, x):
        '''
        Calculate the density of a point x=(x1, x2)
        '''
        return self.fixpdf.pdf(x)+(1/self.c_sum) * np.maximum(1 - (1/self.x_s) * np.abs(x[0] - self.x_mu) - (1/self.y_s) * np.abs(x[1] - self.y_mu), 0)

    def pdf_xlist(self, x):
        '''
        Calculate the density of a point x=(x1, x2)
        '''
        return self.fixpdf.pdf(x)+(1/self.c_sum) * np.maximum(1 - (1/self.x_s) * np.abs(x[:,0] - self.x_mu) - (1/self.y_s) * np.abs(x[:,1] - self.y_mu), 0)

    def rvs_x(self, y):
        '''
        Generate a sample with p(x|y).
        '''
        scale = (self.y_s - np.abs(y - self.y_mu))/self.y_s * self.x_s
        l_center = self.x_mu
        return triang.rvs(c=0.5, loc=l_center-scale, scale=2*scale)

    def rvs_y(self, x):
        '''
        Generate a sample with p(y|x).
        '''
        scale = (self.x_s - np.abs(x - self.x_mu))/self.x_s * self.y_s
        l_center = self.y_mu
        return triang.rvs(c=0.5, loc=l_center-scale, scale=2*scale)
        
def gibbs(target, ini_x, sample_size=10000):
    '''
    Metropolis-Hastings algorithm
        Actually this is Metropolis algorithm.
        Use normal distribution as jump probability.
        ini_x: the initial guess.
        sample_size: the number of samples
    '''
    ini_x = np.array(ini_x)
    res = np.zeros([sample_size, 2])
    for i in range(sample_size):
        res[i,:] = ini_x
        # Gibbs step 1
        y_c = target.rvs_y(ini_x[0])
        x_c = target.rvs_x(y_c)
        ini_x = np.array([x_c, y_c])
    return res

def display_density(pfunc, x_l, x_s):
    '''
    Display the density function in the square range of x_l -> x_s
    '''
    x = np.linspace(x_l[0], x_s[0], 1000)
    y = np.linspace(x_l[1], x_s[1], 1000)
    unit = np.abs((x[1]-x[0])*(y[1]-y[0]))
    X, Y = np.meshgrid(x, y)
    sze = X.shape
    #print(sze)
    Z = pfunc(np.vstack([X.ravel(), Y.ravel()]).T)
    Z.resize(sze)
    #print(Z.shape)
    #plt.plot(np.diff(np.sort(unit*np.cumsum(np.cumsum(Z,0),1).ravel())))
    plt.imshow(Z, extent=[x_l[0], x_s[0], x_s[1], x_l[1]])
    #plt.imshow(unit*np.cumsum(np.cumsum(Z,0),1), extent=[x_l[0], x_s[0], x_s[1], x_l[1]])
    plt.show()

def estise(samples, esti_size=1000, mu=[0,0]):
    N=samples.shape[0]
    mu = np.array(mu)
    res = np.zeros([N-esti_size])
    x = esti_size + np.cumsum(np.ones([N-esti_size]))
    for i in range(0, N-esti_size):
        res[i] = np.mean(np.power(samples[i:i+esti_size] - mu, 2))
    return res, x

def estimu(samples, esti_size=1000, mu=[0,0]):
    N=samples.shape[0]
    mu = np.array(mu)
    res = np.zeros([N-esti_size])
    x = esti_size + np.cumsum(np.ones([N-esti_size]))
    for i in range(0, N-esti_size):
        res[i] = np.sqrt(np.sum(np.power(np.mean(samples[i:i+esti_size,:], axis=0) - mu, 2), axis=0))
    return res, x

if __name__=='__main__':
    t2d = Traingle2D(0.5, 1, 2.0, 1.4)
    rsamples = gibbs(t2d, [0.4, 0.8], sample_size=20000)
    samples = rsamples[-10000:,:]
    display_density(t2d, [-2,-1], [3,3])
    #plt.plot(samples[:,0])
    plt.scatter(x = samples[:,0], y = samples[:, 1], s=1)
    plt.show()
    SE, x = estise(rsamples, mu=[0.5, 1], esti_size=1000)
    plt.plot(x, SE)
    plt.gcf().set_size_inches(10,3)
    plt.show()
    d2mu, x = estimu(rsamples, mu=[0.5, 1], esti_size=1000)
    plt.plot(x, d2mu)
    plt.gcf().set_size_inches(10,3)
    plt.show()