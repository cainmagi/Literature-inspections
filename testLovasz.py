#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Project - Test Lovasz loss function
# Designed by Yuchen Jin (cainmagi)
# Requirements: 
#   python 3.6, numpy
# Version 1.0, Date: 2018/11/9
# Comments:
#   A numpy realization for Lovasz function. This 
#   project is used for simulating the behavior of
#   such an extension of the original set function.
#   Theoretically, the Lovasz extension should be
#   exactly the interpolation of the original set 
#   function.
####################################################
'''
import sys, os
import numpy as np

def oneHotJaccard(logits, labels):
    '''
    Calculate the Jaccard Index between logits and labels.
      logits, labels are [..., c] matrices,
      we use c to represent the channel number.
    '''
    binLogits = np.greater(logits, 0.5)
    binLabels = np.greater(labels, 0.5)
    valNumer = np.logical_and(binLogits, binLabels).astype(np.float32)
    valDomin = np.logical_or(binLogits, binLabels).astype(np.float32)
    c = valNumer.shape[-1]
    valNumer = np.sum(np.reshape(valNumer, [-1, c]), axis=0)
    valDomin = np.sum(np.reshape(valDomin, [-1, c]), axis=0)
    if valNumer == 0.0 and valDomin == 0.0:
        return 0.0
    res = np.divide(valNumer, valDomin)
    return res
    
def oneHotJaccard_non_c(logits, labels, withI=None):
    '''
    Calculate the Jaccard Index between logits and labels.
      logits, labels are vectors.
    '''
    if len(logits) == 0:
        return 1.0
    binLogits = np.greater(logits, 0.5)
    binLabels = np.greater(labels, 0.5)
    valNumer = np.not_equal(binLogits, binLabels)
    if withI is not None:
        valNumer[withI:] = False
    valDomin = np.logical_or(binLabels, valNumer).astype(np.float32)
    valNumer = valNumer.astype(np.float32)
    valNumer = np.sum(valNumer.ravel())
    valDomin = np.sum(valDomin.ravel())
    #print(valDomin)
    if valNumer == 0.0 and valDomin == 0.0:
        return 1.0
    res = np.divide(valNumer, valDomin)
    return res

def lossFunc(logits, labels):
    #misfit = np.abs(logits - labels)
    #misfit = 1-np.cos(np.pi*(logits - labels)/2)
    #misfit = np.square(logits - labels)
    misfit = -labels*np.log10(0.1+logits) - (1-labels) * np.log10(1.1-logits)
    return misfit
    
def hingeLoss(logits, labels):
    return np.maximum(0, 1-logits * labels)
    
def plainLovasz(setfunc, lossfunc, logits, labels):
    '''
	Calculate the Lovasz function by direct computing (theory).
	'''
    logits = logits.astype(np.float32)
    labels = labels.astype(np.float32)
    c = logits.shape[-1]
    logits = np.reshape(logits, [-1, c])
    labels = np.reshape(labels, [-1, c])
    res = np.zeros([c])
    misfit = lossfunc(logits, labels)
    indLen = labels.shape[0]
    for k in range(c):
        misfitk = misfit[:, k]
        misind = np.argsort(-misfitk)
        misfitk = misfitk.copy()[misind]
        getSum = 0
        labelsK = labels[misind,k]
        logitsK = logits[misind,k]
        #print(misind, misfitk, logitsK, labelsK)
        for i in range(indLen):
            gi = setfunc(logitsK, labelsK, i+1) - setfunc(logitsK, labelsK, i)
            getSum = getSum + misfitk[i]*gi
            #print(i, gi, misfitk[i], sep=',  ')
        res[k] = getSum
    return res
    
def JaccardLovasz(lossfunc, logits, labels):
    '''
	Calculate the Lovasz extension for the Jaccard index by improved algorithm.
	'''
    logits = logits.astype(np.float32)
    labels = labels.astype(np.float32)
    c = logits.shape[-1]
    logits = np.reshape(logits, [-1, c])
    labels = np.reshape(labels, [-1, c])
    res = np.zeros([c])
    misfit = lossfunc(logits, labels)
    p = labels.shape[0]
    labels = np.greater(labels, 0.5).astype(np.float32)
    for k in range(c):
        misfitK = misfit[:, k]
        misind = np.argsort(-misfitK)
        misfitK = misfitK.copy()[misind]
        labelsK = labels[misind,k]
        labelsS = np.sum(labelsK)
        intersection = labelsS - np.cumsum(labelsK)
        union = labelsS + np.cumsum(1.0 - labelsK)
        g = 1 - intersection/union
        if p > 1:
            g[1:p] = g[1:p] - g[0:(p-1)]
        res[k] = np.sum(misfitK*g)
    return res
    
def xyz2csv(filename, nparray, x, y):
    with open(filename+'.csv', 'wb') as f:
        f.write(','.encode('utf8'))
        n_row = nparray.shape[0]
        np.savetxt(f, np.reshape(np.arange(3*nparray.shape[1]), (1,3*nparray.shape[1])), fmt='%d', delimiter=',')
        A = np.concatenate([nparray, x, y], axis=1)
        for j in range(n_row):
            f.write('{0:d},'.format(j).encode('utf8'))
            np.savetxt(f, A[j:j+1,:], fmt='%-4.04f', delimiter=',')
            
def multiz2csv(filename, nparrays, x, y):
    with open(filename+'.csv', 'wb') as f:
        f.write(','.encode('utf8'))
        n_row = nparrays[0].shape[0]
        n_col = nparrays[0].shape[1]
        matnum = len(nparrays)
        np.savetxt(f, np.reshape(np.arange((matnum+2)*n_col), (1,(matnum+2)*n_col)), fmt='%d', delimiter=',')
        A = np.concatenate([*nparrays, x, y], axis=1)
        for j in range(n_row):
            f.write('{0:d},'.format(j).encode('utf8'))
            np.savetxt(f, A[j:j+1,:], fmt='%-4.04f', delimiter=',')
            
def xyz2multi(filename, files, matsize):
    matQueue = []
    X = None
    Y = None
    for fname in files:
        with open(fname+'.csv', 'rb') as fi:
            fi.readline()
            getMat = np.loadtxt(fi, delimiter=',')
            matQueue.append(getMat[:,1:(matsize+1)])
            if X is None:
                X = getMat[:,(matsize+1):(2*matsize+1)]
            if Y is None:
                Y = getMat[:,(2*matsize+1):(3*matsize+1)]
    multiz2csv(filename, matQueue, X, Y) 
    
if __name__ == '__main__':
    import scipy.interpolate as scip
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d.axes3d import get_test_data
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    
    def test_LovaszFeature():
        p  = np.array([[0.0],[0.0],[1.0],[1.0],[0.0]])
        gt = np.array([[0.0],[0.0],[1.0],[0.0],[1.0]])
        prec = 50
        meshx = np.linspace(0.0, 1.0, prec)
        meshy = np.linspace(0.0, 1.0, prec) #np.array([0.0, 0.4999, 0.5001, 1.0])
        precy = meshy.shape[0]
        resA = np.zeros([precy, prec])
        resB = np.zeros([precy, prec])
        resC = np.zeros([precy, prec])
        X, Y = np.meshgrid(meshx, meshy)
        for i in range(precy):
            for j in range(prec):
                p[0,0] = X[i,j]
                #p[1,0] = vj
                gt[0,0] = Y[i,j]
                resA[i,j] = JaccardLovasz(lossFunc, p, gt)
                resB[i,j] = plainLovasz(oneHotJaccard_non_c, lossFunc, p, gt)
                resC[i,j] = oneHotJaccard(p, gt)
                #print(resA[i,j], resB[i,j], sep=',  ')
        fig = plt.figure()
        axA = fig.add_subplot(1, 3, 1, projection='3d')
        #axA.imshow(np.nan_to_num(resA, copy=True))
        surfA = axA.plot_surface(X, Y, np.nan_to_num(1-resA, copy=True), rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
        axB = fig.add_subplot(1, 3, 2, projection='3d')
        surfB = axB.plot_surface(X, Y, np.nan_to_num(1-resB, copy=True), rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
        axC = fig.add_subplot(1, 3, 3, projection='3d')
        surfC = axC.plot_surface(X, Y, np.nan_to_num(resC, copy=True), rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
        #axB.imshow(np.nan_to_num(1-resB, copy=True))
        #fig.colorbar(surfB, shrink=0.5, aspect=10)
        plt.show()
        #multiz2csv('lovasz-test-00101-non', [np.nan_to_num(resC, copy=True)], X, Y)
        #multiz2csv('lovasz-comp-11-ent', [np.nan_to_num(1-resA, copy=True), np.nan_to_num(resC, copy=True)], X, Y)
        
    def test_jacLovasz_softmax():
        p  = np.array([[0.0],[0.0]])
        gt = np.array([[1.0],[1.0]])
        prec = 50
        meshx = np.linspace(-5.0, 5.0, prec)
        meshy = np.linspace(-5.0, 5.0, prec)
        x_d = meshx * gt[0,0] - meshx * (1-gt[0,0])
        y_d = meshy * gt[1,0] - meshy * (1-gt[1,0])
        X, Y = np.meshgrid(x_d, y_d)
        X_s, Y_s = np.meshgrid(1 / (1 + np.exp(-meshx)), 1 / (1 + np.exp(-meshy)))
        res = np.zeros([prec, prec])
        for i in range(prec):
            for j in range(prec):
                p[0,0] = X_s[i,j]
                #p[1,0] = vj
                p[1,0] = Y_s[i,j]
                #res[i,j] = plainLovasz(oneHotJaccard_non_c, lossFunc, p, gt)
                res[i,j] = JaccardLovasz(lossFunc, p, gt)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        surf = ax.plot_surface(X, Y, np.nan_to_num(res, copy=True), rstride=1, cstride=1, cmap=cm.coolwarm,
                   linewidth=0, antialiased=True)
        plt.show()
        xyz2csv('lovasz-softmax-00', np.nan_to_num(res, copy=True), X, Y)
        
    def test_jacLovasz_hinge():
        p  = np.array([[0.0],[0.0]])
        gt = np.array([[1.0],[1.0]])
        prec = 50
        x_r = np.linspace(-2.0, 2.0, prec)
        y_r = np.linspace(-2.0, 2.0, prec)
        meshx = (1 - x_r)/ gt[0,0]
        meshy = (1 - y_r) / gt[1,0]
        X, Y = np.meshgrid(x_r, y_r)
        X_s, Y_s = np.meshgrid(meshx, meshy)
        res = np.zeros([prec, prec])
        for i in range(prec):
            for j in range(prec):
                p[0,0] = X_s[i,j]
                #p[1,0] = vj
                p[1,0] = Y_s[i,j]
                #res[i,j] = plainLovasz(oneHotJaccard_non_c, hingeLoss, p, gt)
                res[i,j] = JaccardLovasz(hingeLoss, p, gt)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        surf = ax.plot_surface(X, Y, np.nan_to_num(res, copy=True), rstride=1, cstride=1, cmap=cm.coolwarm,
                   linewidth=0, antialiased=True)
        plt.show()
        xyz2csv('lovasz-hinge-pn', np.nan_to_num(res, copy=True), X, Y)
        
    def test_example():
        prec = 100
        xx = np.linspace(-2.0, 2.0, prec)
        x = np.array([-2, -1, 0, 1, 2])
        y = np.array([0, 3, 1, 2, 0])
        f1 = scip.interp1d(x, y, kind='previous')
        f2 = scip.interp1d(x, y, kind='cubic')
        yy1 = f1(xx)
        yy2 = f2(xx)
        fig = plt.gcf()
        fig.set_size_inches(5,4.5)
        plt.subplot(211)
        plt.plot(x, y, 'o', xx, yy1, '-')
        plt.ylim([-0.2, 3.3])
        plt.ylabel('Binarized')
        plt.subplot(212)
        plt.plot(x, y, 'o', xx, yy2, '-')
        plt.ylabel('Interpolated')
        plt.ylim([-0.2, 3.3])
        plt.show()
        fig = plt.gcf()
        fig.set_size_inches(5,4.5)
        plt.subplot(211)
        plt.plot(xx[:-1], np.zeros(xx.shape[0]-1), '-.', color='black')
        plt.plot(xx[:-1], np.diff(yy1)/(xx[1]-xx[0]), color='C1')
        #plt.ylim([-0.2, 3.3])
        plt.ylabel('Binarized')
        plt.subplot(212)
        plt.plot(xx[:-1], np.zeros(xx.shape[0]-1), '-.', color='black')
        plt.plot(xx[:-1], np.diff(yy2)/(xx[1]-xx[0]), color='C1')
        plt.ylabel('Interpolated')
        #plt.ylim([-0.2, 3.3])
        plt.show()
      
    test_LovaszFeature()   
    #test_jacLovasz_softmax()
    #test_example()
    #xyz2multi('lovasz-softmax', ['lovasz-softmax-00', 'lovasz-softmax-01', 'lovasz-softmax-10', 'lovasz-softmax-11'], 50)