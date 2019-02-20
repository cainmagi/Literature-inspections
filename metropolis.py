#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Project - Test Metropolis-Hastings algorithm and
#           its convergence
# Designed by Yuchen Jin (cainmagi)
# Requirements: 
#   python 3.6+, numpy 1.14+, scipy 1.1+, 
#   matplotlib 3+
# Version 1.0, Date: 2019/02/19
# Comments:
#   A numpy simulation of Metropolis-Hastings alg-
#   orithm. In this code, generate the samples from
#   a 2D distribution by applying MH algorithm. And
#   we would also estimate the standard error.
####################################################
'''
import inspect
import numpy as np
from scipy.stats import multivariate_normal
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
        
def metropolis(target, ini_x, sigma, sample_size=10000, valid_thres=0.01):
    '''
    Metropolis-Hastings algorithm
        Actually this is Metropolis algorithm.
        Use normal distribution as jump probability.
        ini_x: the initial guess.
        sigma: the sigma value of jump distribution
        sample_size: the number of samples
        valid_thres: the minimum of the acceptance rate.
    '''
    ini_x = np.array(ini_x)
    res = np.zeros([sample_size, 2])
    cov_sig = np.diag([sigma, sigma])
    norm_rv = multivariate_normal([0,0], cov_sig)
    valid = np.zeros([sample_size])
    for i in range(sample_size):
        res[i,:] = ini_x
        # generate candidate
        x_c = ini_x + norm_rv.rvs()
        p1 = target(x_c)
        p2 = target(ini_x)
        if p2 > valid_thres:
            valid[i] = 1.0
        #print(p1/p2, end=' ')
        if p1 < valid_thres:
            alpha = 0.0
        else:
            alpha = min(1, p1/p2)
        r = np.random.rand()
        if r < alpha:
            ini_x = x_c
    return res, valid

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

if __name__=='__main__':
    t2d = Traingle2D(0.5, 1, 2.0, 1.4)
    rsamples, valid = metropolis(t2d, [0.4, 0.8], 1, sample_size=20000)
    samples = rsamples[-10000:,:]
    valid = valid[-10000:]
    display_density(t2d, [-2,-1], [3,3])
    #plt.plot(samples[:,0])
    if np.all(valid > 0.6):
        scmap=ListedColormap([[0,0,1]])
    else:
        scmap=ListedColormap([[1,0,0],[0,0,1]])
    plt.scatter(x = samples[:,0], y = samples[:, 1], s=1, c=valid, cmap=scmap)
    plt.show()
    SE, x = estise(rsamples, mu=[0.5, 1])
    plt.plot(x, SE)
    plt.gcf().set_size_inches(10,3)
    plt.show()