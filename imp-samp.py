#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Project - Test error of self-normalized importance
#           sampling
# Designed by Yuchen Jin (cainmagi)
# Requirements: 
#   python 3.6+, numpy 1.145
# Version 1.0, Date: 2019/02/18
# Comments:
#   A numpy simulation of importance sampling. In 
#   this code, we would compare:
#     a. real expectation
#     b. the simulation of the unbiased estimator 
#        for the importance sampling
#     c. the simulation of biased estimator for the
#        self-normalized importance sampling
####################################################
'''
import numpy as np

def density_p(x):
    '''
    Calculate the density of a point x=(x1, x2)
    '''