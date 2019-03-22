# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:47:24 2019

@author: gela
"""




'''


VERY EASILY READABLE CODE

'''


import numpy as np
from Nafo.cnn_theano import Convoutional_Neural_net as cnn
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as T

data = np.array(pd.read_csv('train.csv')).astype(np.float32)

Y_train = data[:1000,0].astype(np.int32)
X_train = data[:1000, 1:]

X_reshaped = np.float32(X_train.reshape(1000, 1, 28, 28)/255)

'''
plt.imshow(X_reshaped[3, 0, :, :], cmap = 'Greys')
plt.show()
'''

gela = cnn(CNN = [((5, 1, 5, 5), T.nnet.relu, (2,2)),
                  ((10, 5, 5, 5), T.nnet.relu, (2,2))
                 ],
           ANN = [(('x', 700), T.nnet.relu),
                  ((700, 500), T.nnet.relu),
                  ((500, 10), T.nnet.softmax)],
           shape_x = 28)


gela.load_from_files()
pred = gela.predict(X = X_reshaped, Y=  Y_train)


'''

works like charm

'''
