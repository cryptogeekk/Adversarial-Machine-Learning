#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:07:29 2021

@author: krishna
"""
from tensorflow import keras
import pandas as pd
import numpy as np

#changes in a local branch

class mnist_poison:
    def __init__(self,percent,poisoned_labels,dataset_y):
        self.percent=percent
        self.poisoned_labels=poisoned_labels
        self.dataset=dataset_y
        
    def poison_validity_check(self,previous_dataset, poisoned_dataset):
        previous_dataset=pd.DataFrame(previous_dataset)
        validation_count=0
        for index in range(len(previous_dataset)):
            if previous_dataset[0][index]!=poisoned_dataset[0][index]:
                validation_count+=1
        print("The number of data poisoned is ",validation_count)
        
    def poison(self):
        #variables for temporary purpose
        # percent=10
        # poisoned_labels=[0,1,2]
        
        # fashion_mnist = keras.datasets.mnist
        # (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        y_train_full=pd.DataFrame(self.dataset)
        labels=y_train_full[0].unique()
        
        poison_amount=int((self.percent/100)*len(y_train_full))
        
        y_train_full.count()
        y_train_temp=y_train_full.copy()
        
        if len(self.poisoned_labels)==0:
            for index in range(poison_amount):
                random_index=np.random.randint(0,9)
                
                for i in range(0,100):
                    if y_train_full[0][index]==labels[random_index]:
                        random_index=np.random.randint(0,9)
                    else:
                        break
                    
    
                y_train_full[0][index]=labels[random_index]
                print("MNIST Data withOUT defined labels "+ "poisoned witha amount =",poison_amount)
                        
        else:
            count=1
            for index in range(len(y_train_full)):
                if y_train_full[0][index] in self.poisoned_labels:                    
                    random_index=np.random.randint(0,9)
                    
                    for i in range(0,100):
                        if y_train_full[0][index]==labels[random_index]:
                            random_index=np.random.randint(0,9)
                        else:
                            break
                        
                    y_train_full[0][index]=labels[random_index]
                    count=count+1
                    
                if count==poison_amount:
                    print("MNIST Data with defined labels "+ "poisoned witha amount =",poison_amount)
                    break
        
        return y_train_full
                
                    
                    

               
                    
      
                      
              
