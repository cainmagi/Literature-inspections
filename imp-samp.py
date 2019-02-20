#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Project - Test error of self-normalized importance
#           sampling
# Designed by Yuchen Jin (cainmagi)
# Requirements: 
#   python 3.6+, numpy 1.14+, scipy 1.1+, 
#   matplotlib 3+
# Version 1.0, Date: 2019/02/18
# Comments:
#   A numpy simulation of importance sampling. In 
#   this code, we would compare:
#     a. real expectation
#     b. the simulation of the unbiased estimator 
#        for the importance sampling
#     c. the simulation of biased estimator for the
#        self-normalized importance sampling
#   The test is performed by using normal distribu-
#   tion to sample chi-squared distribution.
####################################################
'''
import inspect
import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
import matplotlib.pyplot as plt

def gen_chisquare(func, k, xrange=None):
    '''
    Generate a distribution over chi-squared distribution.
        func: an arbitrary single-input function.
        k: degree of freedom in chi-squared distribution.
        xrange: the testing range, if set 0, generated automatically.
    '''
    if xrange is None:
        xrange = [chi2.ppf(0.001, k), chi2.ppf(0.999, k)]
    #mean, var, skew, kurt = chi2.stats(k, moments='mvsk')
    x = np.linspace(*xrange, 1000)
    y = func(x) * chi2.pdf(x, k)
    return x, y

def gen_normal(func, mu, sigma, xrange=None):
    '''
    Generate a distribution over normal distribution.
        func: an arbitrary single-input function.
        mu, sigma: the expectation and standard deviation.
        xrange: the testing range, if set 0, generated automatically.
    '''
    if xrange is None:
        xrange = [norm.ppf(0.001, mu, sigma), norm.ppf(0.999, mu, sigma)]
    #mean, var, skew, kurt = chi2.stats(k, moments='mvsk')
    x = np.linspace(*xrange, 1000)
    y = func(x) * norm.pdf(x, mu, sigma)
    return x, y

def draw_pdf(genfunc, *args, **kwargs):
    '''
    Draw probability density function of a distribution.
        genfunc: the function by which we generate the distribution.
        [args]: the arguments passed to the genfunc(lambda x:1, *args).
    '''
    # Get the arguments for plotting
    LineName = kwargs.pop('LineName', None)
    # Clean the unused keywords
    arglist = inspect.getfullargspec(genfunc)[0]
    poplist = []
    for arg in kwargs.keys():
        if not arg in arglist:
            poplist.append(arg)
    for arg in poplist:
        kwargs.pop(arg)
    # Perform the function
    x, y = genfunc(lambda x:1, *args, **kwargs)
    plt.plot(x, y, label=LineName)

def sampling_test(targetfunc, chi2_k=10, norm_mu=20, norm_sigma=10, sample_size=10000, figureSize=(6,4)):
    '''
    Test the importance sampling
    '''
    chi2At = chi2(chi2_k)
    normAt = norm(norm_mu, norm_sigma)
    chi2_x = chi2At.rvs(size=sample_size)
    norm_x = normAt.rvs(size=sample_size)
    # Original sampling
    chi2_y = targetfunc(chi2_x)
    # Importance sampling
    weights = chi2At.pdf(norm_x) / normAt.pdf(norm_x)
    norm_y = targetfunc(norm_x) * weights
    # Calculate the Monte-Carlo simulation for mu
    chi2_rmu  = np.cumsum(chi2_y) / np.cumsum(np.ones(chi2_y.shape))
    norm_rmu  = np.cumsum(norm_y) / np.cumsum(np.ones(norm_y.shape))
    norm_rmuW = np.cumsum(norm_y) / np.cumsum(weights)
    # Calculate the real integration
    unit, real_rmu = gen_chisquare(targetfunc, chi2_k, xrange=[-100, 1000])
    unit = unit[1]-unit[0]
    real_rmu = np.sum(real_rmu) * unit
    # Plot the estimations for expectation
    plt.plot(chi2_rmu, label='estimation for $E_q[f(x)]$')
    plt.plot(norm_rmu, label='estimation for $E_p[f(x)q(x)/p(x)]$')
    plt.plot(norm_rmuW, label='estimation for $E_p[wf(x)/E_p(w]$')
    plt.plot([1, sample_size], [real_rmu, real_rmu], label='real $E_q[f(x)]$', color='red', linestyle='--')
    plt.legend()
    showrange = max(1, 1.1*np.mean(np.abs(chi2_rmu[-5:]-real_rmu)), 1.1*np.mean(np.abs(norm_rmu[-5:]-real_rmu)), 1.1*np.mean(np.abs(norm_rmuW[-5:]-real_rmu)))
    plt.ylim([real_rmu-showrange, real_rmu+showrange])
    plt.gcf().set_size_inches(*figureSize)
    plt.show()
    # Calculate the Monte-Carlo simulation for variance
    chi2_Y, chi2_YB = np.meshgrid(chi2_y, chi2_rmu)
    chi2_Y = np.sum(np.power((chi2_Y - chi2_YB) * np.tri(chi2_Y.shape[0], chi2_Y.shape[0]), 2), axis=1)
    chi2_rsig = chi2_Y[1:] / np.cumsum(np.ones(chi2_Y.shape[0]-1))
    del chi2_YB
    norm_Y, norm_YB = np.meshgrid(norm_y, norm_rmu)
    norm_Y = np.sum(np.power((norm_Y - norm_YB) * np.tri(norm_Y.shape[0], norm_Y.shape[0]), 2), axis=1)
    norm_rsig = norm_Y[1:] / np.cumsum(np.ones(norm_Y.shape[0]-1))
    del norm_YB
    # Calculate the real variance
    unit, real_rsig = gen_chisquare(lambda x:np.power(targetfunc(x)-real_rmu, 2), chi2_k, xrange=[-100, 1000])
    unit = unit[1]-unit[0]
    real_rsig = np.sum(real_rsig) * unit
    # Plot the estimations for variance
    plt.plot(chi2_rsig, label='estimation for $D_q[f(x)]$')
    plt.plot(norm_rsig, label='estimation for $D_p[f(x)q(x)/p(x)]$')
    plt.plot([1, sample_size-1], [real_rsig, real_rsig], label='real $D_q[f(x)]$', color='red', linestyle='--')
    plt.legend()
    plt.ylim([ real_rsig/2, max([real_rsig*3, 1.1*norm_rsig[-1], 1.1*chi2_rsig[-1]]) ])
    plt.gcf().set_size_inches(*figureSize)
    plt.show()
    # Calculate the biased estimation for standard error
    norm_YW, norm_YBW = np.meshgrid(weights*targetfunc(norm_x), weights*norm_rmuW)
    norm_YW = np.sum(np.power((norm_YW - norm_YBW) * np.tri(norm_YW.shape[0], norm_YW.shape[0]), 2), axis=1)
    norm_seW = norm_Y[1:] / np.power(np.cumsum(weights)[1:],2)
    # Plot the estimations for standard error
    chi2_se = norm_rsig/(np.cumsum(norm_rsig)+1)
    norm_se = norm_rsig/(np.cumsum(norm_rsig)+1)
    plt.plot(chi2_se, label='estimation for $SE^2[f(x)]$')
    plt.plot(norm_se, label='estimation for $SE^2[f(x)q(x)/p(x)]$')
    plt.plot(norm_seW, label='estimation for $SE^2$ (biased)')
    plt.legend()
    plt.ylim([-0.01, max(0.3, 3*np.mean(chi2_se[-5:]), 3*np.mean(norm_se[-5:]), 3*np.mean(norm_seW[-5:]))])
    plt.gcf().set_size_inches(*figureSize)
    plt.show()

if __name__=='__main__':
    # Recommend that
    # mu1 = 20, sigma1=10
    # mu2 = 9, sigma2=6
    mu = 20
    sigma = 10
    draw_pdf(gen_chisquare, k=10, xrange=[-10, 50], LineName='Chi-squared')
    draw_pdf(gen_normal, mu=mu, sigma=sigma, xrange=[-10, 50], LineName='Normal')
    plt.legend()
    plt.gcf().set_size_inches(6, 4)
    plt.show()
    sampling_test(
        lambda x:x, 
        chi2_k=10, 
        norm_mu=mu, 
        norm_sigma=sigma, 
        sample_size=10000,
        figureSize=(6,4)
    )
    #sampling_test(lambda x:np.exp(-0.5*np.power(x-5,2)))