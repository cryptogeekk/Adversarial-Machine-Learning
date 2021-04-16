#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:07:29 2021

@author: krishna
"""
class poison:
    def __init__(self,percent,poisoned_labels,dataset):
        self.percent=percent
        self.posioned_labels=poisoned_labels
        self.dataset=dataset
        
    def poison():
        fashion_mnist = keras.datasets.mnist
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
        
        
        

