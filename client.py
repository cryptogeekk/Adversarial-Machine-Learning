#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:43:56 2021

@author: krishna
"""
class Client:
    
    def __init__(self, dataset_x, dataset_y, epoch_number, learning_rate):
        self.dataset_x=dataset_x
        self.dataset_y=dataset_y
        # self.mini_batch=mini_batch
        self.epoch_number=epoch_number
        self.learning_rate=learning_rate
        # self.decay_rate=decay_rate
        
    def train(self):
        self.dataset_x=self.dataset_x/255.0
        self.dataset_y=self.dataset_y/255.0
        
        import numpy as np
        import pandas as pd
        import matplotlib as plt
        from tensorflow import keras
        
        model=keras.models.Sequential([
                keras.layers.Flatten(input_shape=[28,28]),
                keras.layers.Dense(200,activation='relu'),
                keras.layers.Dense(200,activation='relu'),
                keras.layers.Dense(10,activation='softmax')
            ])
        
        #getting the initial weight of the model
        initial_weight=model.get_weights()
        # output_weight_list=[]
        
        #training the model
        # import animation
        print('###### Client1 Training started ######')
        # wait=animation.Wait()
        # wait.start()
        
        model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=self.learning_rate),metrics=['accuracy'])
        history=model.fit(self.dataset_x, self.dataset_y,epochs=self.epoch_number) 
        
        #getting the final_weight
        output_weight=model.get_weights()
        
        # wait.stop()
        
        return output_weight
        
        
        
        
        
        



    