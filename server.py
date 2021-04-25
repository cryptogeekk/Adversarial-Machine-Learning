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

#loading the dataset
fashion_mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


import dataset_divider
x_data, y_data=dataset_divider.divide_with_label(5,X_train, y_train)
x_data, y_data=dataset_divider.get_data[x_data,y_data,4,'non-iid']

#function for averaging
def model_average(client_weights):
    average_weight_list=[]
    for index1 in range(len(client_weights[0])):
        layer_weights=[]
        for index2 in range(len(client_weights)):
            weights=client_weights[index2][index1]
            layer_weights.append(weights)
        average_weight=np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_list.append(average_weight)
    return average_weight_list
            

def create_model():
    model=keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        keras.layers.Dense(200,activation='relu'),
        keras.layers.Dense(200,activation='relu'),
        keras.layers.Dense(10,activation='softmax')
    ])
    
    weight=model.get_weights()
    return weight
    
    
    
#initializing the client automatically
from client import Client
def train_server(training_rounds,epoch,learning_rate):
    # client_weights=[]
    accuracy_list=[]
    
    #temp_variable
    training_rounds=50
    epoch=5 
    learning_rate=0.1
    client_weight_for_sending=[]
    
    for index1 in range(1,training_rounds):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged=[]
        for index in range(len(y_data)):
            print('-------Client-------', index)
            if index1==1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight=create_model()
                client=Client(x_data[index],y_data[index],epoch,learning_rate,initial_weight)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
            else:
                client=Client(x_data[index],y_data[index],epoch,learning_rate,client_weight_for_sending[index1-2])
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
    
        #calculating the avearge weight from all the clients
        client_average_weight=model_average(client_weights_tobe_averaged)
        client_weight_for_sending.append(client_average_weight)


        #validating the model with avearge weight
        model=keras.models.Sequential([
                keras.layers.Flatten(input_shape=[28,28]),
                keras.layers.Dense(200,activation='relu'),
                keras.layers.Dense(200,activation='relu'),
                keras.layers.Dense(10,activation='softmax')
            ])

        model.set_weights(client_average_weight)
        model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.1),metrics=['accuracy'])
        result=model.evaluate(X_valid, y_valid)
        accuracy=result[1]
        print('#######-----Acccuracy for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list.append(accuracy)
        return accuracy_list

training_accuracy=train_server(50,5,0.1)


#initializing the client manually
client1_train_x, client1_train_y=dataset_divider.get_data(x_data, y_data, 0,'non--iid')
from client import Client
client1=Client(client1_train_x,client1_train_y,30,0.1)
client1_weight=client1.train()


#temp work
temp_list=[]
for index in range(len(client_weights)):
    temp_list.append(client_weights[index][5])
    print(client_weights[index][5])
    
    
list=[1,2,3,4,5]
np.average(list)

temp=np.mean(np.array([client_weights[0][5], client_weights[1][5]]), axis=0)

temp=np.mean(np.array([x for x in temp_list]), axis=0)

list5=[np.array([1,2,3]),np.array([4,5,6])]
temp_mean=np.mean(np.array([list5[0], list5[1]]), axis=0)


