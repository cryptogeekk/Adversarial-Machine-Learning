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
client1_train_x, client1_train_y=dataset_divider.get_data(x_data, y_data, 0,'non--iid')

from client import Client
client1=Client(client1_train_x,client1_train_y,30,0.1)
client1_weight=client1.train()



