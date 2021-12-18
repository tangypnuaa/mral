#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import math
import os

import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_sylvester
from sklearn.metrics import balanced_accuracy_score
import pandas as pd


lamda = 1
mu = 1
epsilon = 10 ** (-4)


def initialize(k, c):
    d, m = WS.shape
    D = np.mat(np.random.rand(d, k))
    VS = np.mat(np.random.rand(k, m))
    VT = np.mat(np.random.rand(k, c))
    return D, VT, VS


def norm21(W):
    s = 0
    for i in range(np.shape(W)[0]):
        s += norm(W[i], 2)
    return s


def targetf0(D, VS):
    A = norm(WS - D * VS, 'fro') ** 2  + norm21(VS.T)
    return A


def updateVS(D):
    JS = float("inf")
    d,k = D.shape
    m = WS.shape[1]
    SGMS = np.mat(np.eye(m))
    while 1:
        VS = solve_sylvester(2*D.T*D,SGMS,2*D.T*WS)
        for i in range(m):
            SGMS[i, i] = norm(VS[:,i] + 0.000001) ** (-1)
        NJS = targetf0(D, VS)
        if abs(JS - NJS) / NJS < 0.001:
            break
        JS = NJS
    return np.mat(VS)


def learn_dictionary(D, VS):
    J = np.inf
    VS = updateVS(D)
    D = WS * VS.T * (VS * VS.T).I
    NJ = targetf0(D, VS)
    turns = 0
    while abs(J - NJ) / NJ > epsilon or turns < 10:
        if abs(J - NJ) / NJ <= epsilon:
            turns += 1
        else:
            turns = 0
        J = NJ
        VS = updateVS(D)
        D = WS * VS.T * (VS * VS.T).I
        NJ = targetf0(D, VS)
    return D, VS


def targetf(D, VT, WT):
    n = X.shape[0]
    n=2
    A = norm(X * WT - Y, 'fro') ** 2/n + lamda * norm(WT, 'fro')**2 + mu * (norm(WT - D * VT, 'fro') ** 2 + norm21(VT.T))
    return A


def updateVT(D, WT):
    JT = float("inf")
    d,k = D.shape
    c = WT.shape[1]
    SGMT = np.mat(np.eye(c))
    while 1:
        VT = solve_sylvester(2*D.T*D,SGMT,2*D.T*WT)
        for i in range(c):
            SGMT[i, i] = norm(VT[:,i] + 0.000001) ** (-1)
        NJT = norm(D * VT - WT, 'fro') ** 2 + norm21(VT.T)
        if abs(JT - NJT) / NJT < 0.001:
            break
        JT = NJT
    return np.mat(VT)


def optimize(D, VT):
    J = np.inf
    n, d = X.shape
    n=2
    E = np.mat(np.eye(d))
    WT = (X.T * X/n + (lamda + mu) * E).I * (mu * D * VT + X.T * Y/n)
    VT = updateVT(D, WT)
    NJ = targetf(D, VT, WT)
    turns = 0
    while abs(J - NJ) > epsilon or turns < 10:
        if abs(J - NJ) <= epsilon:
            turns += 1
        else:
            turns = 0
        J = NJ
        WT = (X.T * X/n + (lamda + mu) * E).I * (mu * D * VT + X.T * Y/n)
        VT = updateVT(D, WT)
        NJ = targetf(D, VT, WT)
    return VT, WT


def predict(data, WT):
    '''
    data : matrix n*d
    return : ndarry n
    '''
    global ans
    p = []
    for i in data:
        t = (i * WT).tolist()[0]
        p.append(t.index(max(t)) + ans)
    return np.array(p)


def predict_w(WT, data):
    '''
    data : matrix n*d
    return : ndarry n
    '''
    ans = 1
    p = []
    for i in data:
        t = (i * WT).tolist()[0]
        p.append(t.index(max(t)) + ans)
    return np.array(p)


def score(data, label, WT):
    '''
    data : matrix n*d
    label : ndarry n
    '''
    return balanced_accuracy_score(label, predict(data, WT))

def fit(data, label, ws, LAMBDA=1, MU=1, k=10):
    '''
    data : matrix n*d
    label : ndarry n
    ws : matrix d*m
    '''
    global d
    global c
    global m
    global X
    global Y
    global WS
    global lamda
    global mu
    global WT
    global ans
    lamda, mu = LAMBDA, MU
    ans = min(label)
    label = pd.get_dummies(label)
    label = np.mat(label)
    X, Y, WS = data, label, ws
    d = data.shape[1]
    c = label.shape[1]
    m = WS.shape[1]
    D, VT, VS = initialize(k,c)
    D, VS = learn_dictionary(D, VS)
    VT, WT = optimize(D, VT)
    return D, VT, VS, WT
