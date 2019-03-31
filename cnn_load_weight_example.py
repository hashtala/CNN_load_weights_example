# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 21:47:24 2019

@author: gela
"""




'''


SO IF YOU ARE NOT ME I AM NOT SURE IF YOU CAN READ THE CODE...


'''


import numpy as np
from Nafo.cnn_theano import Convoutional_Neural_net as cnn
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as T

data = np.array(pd.read_csv('digits.csv')).astype(np.float32)

Y_train = data[:1000,1].astype(np.int32)
X_train = data[:1000, 2:]

X_reshaped = np.float32(X_train.reshape(1000, 1, 28, 28)/255)

'''
plt.imshow(X_reshaped[3, 0, :, :], cmap = 'Greys')
plt.show()
'''

gela = cnn(CNN = [((30, 1, 5, 5), T.nnet.relu, (2,2)),
                  ((10, 30, 5, 5), T.nnet.relu, (2,2))
                 ],
           ANN = [(('x', 800), T.nnet.relu),
                  ((800, 800), T.nnet.relu),
                  ((800, 10), T.nnet.softmax)],
           shape_x = 28)


gela.load_from_files()



pred = gela.predict(X = X_reshaped, Y=  Y_train)
plt.imshow(X_reshaped[np.random.randint(0,1000), 0, :, :])
plt.show()

'''

works like charm

'''
