import os

import numpy as np
from sklearn.metrics import balanced_accuracy_score
import sklearn
import TMR as tf
import TMR_s as stf
from utils import train_plain_target_model, get_proba_pred, calc_linear_entropy

a = ['a', 'c', 'd', 'w']
W = []
for i in a:
    W.append(np.load(os.path.join('example', 'WS_%s.npy' % i)))
WS = np.vstack((W[1], W[2], W[3])).T
X = np.mat(np.load(os.path.join('example', 'X.npy'))).T
Y = np.transpose(np.load(os.path.join('example', 'Y.npy')))
Y = np.array([i.tolist().index(max(i)) + 1 for i in Y])
vd = np.load(os.path.join('example', 'testdata.npy'))
vdl = np.load(os.path.join('example', 'testlabel.npy'))

tf.fit(X, Y, WS, LAMBDA=1, MU=10, ETA=10, k=19)
print(tf.score(vd, vdl))

_, _, _, Wt = stf.fit(X, Y, WS, LAMBDA=1, MU=10, k=19)
print(stf.score(vd, vdl))

# obtain source models
W = train_plain_target_model(X, Y, 1)
# print(W)
print(W.shape)
# print(Wt)
print(Wt.shape)

ppred = get_proba_pred(W, X)
print(ppred)
for pp in ppred:
    print(calc_linear_entropy(pp))
