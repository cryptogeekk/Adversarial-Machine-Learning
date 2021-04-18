#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 15:46:35 2021

@author: krishna
"""


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd



fashion_mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_names[y_train[0]]

model=keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        # keras.layers.Dense(300,activation='relu'),
        keras.layers.Dense(100,activation='relu'),
        keras.layers.Dense(10,activation='softmax')
    
        ])


model.summary() 
# keras.utils.plot_model(model)

#getting weight of the model
weights,biases=model.layers[2].get_weights()
model.predict(X_train[1])
model_weight=model.get_weights()

#compiling a model
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.1),metrics=['accuracy'])
history=model.fit(X_train, y_train,epochs=10,validation_data=(X_valid,y_valid)) 

#calculating the mean of weights of final layer
mean_list=[]
weights_dataframe=pd.DataFrame(weights)
i=0
for i in range(0,100):
    mean_list.append(weights_dataframe[i].mean())
    i=i+1

#plotting the weights
from pandas.plotting import radviz,lag_plot
radviz(mean_list,'mean_weight')
lag_plot(mean_list)

y=np.arange(0,10)
plt.scatter(y,mean_list)


#creating another model and copying the same weights






import numpy as np
import matplotlib.pyplot as plt

x = np.array([.4,.8,1.2,1.6,2.0,2.4])
y = np.array([.1,.2,.3,.7,.6,.5])

lab = np.array([1,1,2,2,1,2])

for l in np.unique(lab):
    indices = (lab == l)
    plt.scatter(x[indices],y[indices], label=str(l))
plt.legend()
plt.show()





















    
    
model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.67),metrics=['accuracy'])

history=model.fit(X_train, y_train,epochs=20,validation_data=(X_valid,y_valid))
weights,biases=model.layers[3].get_weights()
accuracy=history.history['accuracy']
accuracy[0]=0

communication_round=np.arange(1,31)
plt.plot(communication_round,accuracy)
plt.grid(True)
plt.gca().set_ylim(0,1)        #setting y_limit
plt.show()









#poisoning the model
import temp
fashion_mnist = keras.datasets.mnist
(X_train_full, y_train), (X_test, y_test) = fashion_mnist.load_data()      
original_dataset=y_train.copy()    
poisoned_dataset=temp.poison(10,[7,8,9],y_train)
temp.poison_validity_check(original_dataset,poisoned_dataset)

#dividing the dataset
import dataset_divider
# x_data,y_data=dataset_divider.divide(6,np.arange(1,21),np.arange(21,41))
x_data,y_data=dataset_divider.divide_with_label(1,X_train_full,y_train)
