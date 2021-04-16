#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:07:29 2021

@author: krishna
"""

from tensorflow import keras
import pandas as pd
import numpy as np


        
def poison_validity_check(previous_dataset, poisoned_dataset):
    previous_dataset=pd.DataFrame(previous_dataset)
    poisoned_dataset=pd.DataFrame(poisoned_dataset)
    validation_count=0
    for index in range(len(previous_dataset)):
        if previous_dataset[0][index]!=poisoned_dataset[0][index]:
            validation_count+=1
    print(validation_count)
        
def poison(percent,poisoned_labels,poisoned_dataset):
    y_train_full=pd.DataFrame(poisoned_dataset)
    labels=y_train_full[0].unique()
    
    poison_amount=int((percent/100)*len(y_train_full))
    
    y_train_full.count()
    y_train_temp=y_train_full.copy()
    
    if len(poisoned_labels)==0:
        for index in range(poison_amount):
            random_index=np.random.randint(0,9)
            
            for i in range(0,100):
                if y_train_full[0][index]==labels[random_index]:
                    random_index=np.random.randint(0,9)
                else:
                    break

            y_train_full[0][index]=labels[random_index]
    
        print("MNIST Data withOUT defned labels "+ "poisoned witha amount =",poison_amount)
            
    else:
        count=1
        for index in range(len(y_train_full)):
            if y_train_full[0][index] in poisoned_labels:                    
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
  





                
                    
                    

               
                    
      
                      
              
