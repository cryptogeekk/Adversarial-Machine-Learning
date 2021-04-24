#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:08:06 2021

@author: krishna
"""
import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow import keras

fashion_mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

import dataset_divider
x_data, y_data=dataset_divider.divide_without_label(5,X_train_full, y_train_full)

#function for averaging
def model_average(client_weights):
    average_weight_list=[]
    for index1 in client_weights:
        layer_weights=[]
        for index2 in client_weights[0]:
            weights=array(client_weights[index2][index1])
            layer_weights.append(weights)
            average_weight=average(layer_weights)
            average_weight_list.append(average_weight)

#initializing the client automatically
client_weights=[]
from client import Client

for index in range(1,training_rounds):
    print('Training for round ', index, 'started')
    for index in range(len(y_data)):
        client=Client(x_data[index],y_data[index],10,0.1)
        weight=client.train()
        client_weights.append(weight)
    
    


#initializing the client manually
client1_train_x, client1_train_y=dataset_divider.get_data(x_data, y_data, 0,'non--iid')
from client import Client
client1=Client(client1_train_x,client1_train_y,30,0.1)
client1_weight=client1.train()


#temp work
for index in range(len(client_weights)):
    print(client_weights[index][5])
    
    
list=[1,2,3,4,5]
np.average(list)

average(client1_weight[])