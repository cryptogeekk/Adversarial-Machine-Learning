#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:24:53 2021

@author: krishna
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd


fashion_mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#normalizing the dataset
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# splitting the datatset
import dataset_divider
x_train_data,y_train_data=dataset_divider.divide(10,X_train, y_train)


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model=keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        # keras.layers.Dense(300,activation='relu'),
        keras.layers.Dense(100,activation='relu'),
        keras.layers.Dense(10,activation='softmax')
    
        ])

#getting weights
wsave=model.get_weights()

model.summary() 
# keras.utils.plot_model(model)

output_weight_list=[]

def train_model(X_train,y_train):
    model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.1),metrics=['accuracy'])
    history=model.fit(X_train, y_train,epochs=30,validation_data=(X_valid,y_valid)) 
    
    #getting the weight of the model
    model_weight=model.get_weights()
    output_weight_list.append(model_weight[len(model_weight)-1])
    model.set_weights(wsave)
    
for index in range(len(x_train_data)):
    print('----Going for round-----', index)
    train_model(x_train_data[index],y_train_data[index])
    
    