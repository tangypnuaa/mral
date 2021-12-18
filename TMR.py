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
eta = 1
epsilon = 10 ** (-4)


def initialize(k, c):
    d, m = WS.shape
    D = np.mat(np.random.rand(d, k))
    VS = np.mat(np.random.rand(k, m))
    VT = np.mat(np.random.rand(k, c))
    return D, VT, VS


def norm21(W):
    s = 0
    for i in range(W.shape[0]):
        s += norm(W[i], 2)
    return s


def targetf(D, VS, VT, WT):
    n = X.shape[0]
    A = eta * (norm(WS - D * VS, 'fro') ** 2 + norm21(VS.T)) + norm(X * WT - Y, 'fro') ** 2 / \
        n + lamda * norm(WT, 'fro') ** 2 + mu * \
        (norm(WT - D * VT, 'fro') ** 2 + norm21(VT.T))
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
        NJS = norm(WS - D * VS, 'fro') ** 2 + norm21(VS.T)
        if abs(JS - NJS) / NJS < 0.001:
            break
        JS = NJS
    return np.mat(VS)


def updateVT(D, WT):
    JT = float("inf")
    d,k = D.shape
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


def optimize(D, VT, VS):
    J = np.inf
    n,d = X.shape
    E = np.mat(np.eye(d))
    WT = (X.T*X/n + (lamda+mu) * E).I*(mu*D*VT+X.T*Y/n)
    VS = updateVS(D)
    VT = updateVT(D, WT)
    D = (eta * WS * VS.T + mu * WT * VT.T) * (eta * VS * VS.T + mu * VT * VT.T).I
    NJ = targetf(D, VS, VT, WT)
    turns = 0
    while abs(J - NJ) > epsilon or turns < 10:
        if abs(J - NJ) <= epsilon:
            turns += 1
        else:
            turns = 0
        J = NJ
        WT = (X.T*X/n + (lamda+mu) * E).I*(mu*D*VT+X.T*Y/n)
        VS = updateVS(D)
        VT = updateVT(D, WT)
        D = (eta * WS * VS.T + mu * WT * VT.T) * (eta * VS * VS.T + mu * VT * VT.T).I
        NJ = targetf(D, VS, VT, WT)
    return D, VT, VS, WT


def predict(data):
    '''
    data : matrix n*d
    return ndarry n
    '''
    p = []
    for i in data:
        t = (i * WT).tolist()[0]
        p.append(t.index(max(t)) + ans)
    return np.array(p)


def score(data, label):
    '''
    data : matrix n*d
    label : ndarry n
    '''
    return balanced_accuracy_score(label,predict(data))

def fit(data, label, ws, LAMBDA=1, MU=1, ETA=1, k=10):
    '''
    data : matrix n*d
    label : ndarry n
    ws : matrix d*m
    '''
    global X
    global Y
    global WS
    global lamda
    global mu
    global eta
    global WT
    global ans
    global d
    global c
    global m
    lamda, mu, eta = LAMBDA, MU, ETA
    ans = min(label)
    label = pd.get_dummies(label)
    label = np.mat(label)
    X, Y, WS = data, label, ws
    d = data.shape[1]
    c = label.shape[1]
    m = WS.shape[1]
    D, VT, VS = initialize(k,c)
    D, VT, VS, WT = optimize(D, VT, VS)

