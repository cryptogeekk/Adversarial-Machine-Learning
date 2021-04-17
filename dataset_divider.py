from tensorflow import keras
import pandas as pd
import numpy as np

def divide_without_label(parts, X_train_full,y_train_full):
      
    each_part_number=int(len(X_train_full)/parts)
    list_x_train=[]
    list_y_train=[]
    
    number_list=[]
    number_list.append(0)
    for x in range(1, parts+1):
        number_list.append(each_part_number*x)
    
    
    for x in range(0,parts):
        data_x=X_train_full[number_list[x]:number_list[x+1]]
        data_y=y_train_full[number_list[x]:number_list[x+1]]
        list_x_train.append(data_x)
        list_y_train.append(data_y)
        
    return list_x_train, list_y_train


def divide_with_label(parts, X_train_full, y_train_full):
    fashion_mnist = keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    parts=10
    
    value_counts=pd.Series(y_train_full).value_counts()
    each_part_number=int(len(value_counts)/parts)
    labels=pd.Series(y_train_full).unique()
    
    x_train_list=[[],[],[],[],[],[],[],[],[],[]]
    y_train_list=[[],[],[],[],[],[],[],[],[],[]]
    
    for index in range(len(y_train_full)):
        for index1 in range(len(labels)):
            if y_train_full[index]==labels[index1]:
                y_train_list[labels[index1]].append(y_train_full[index])
                x_train_list[labels[index1]].append(X_train_full[index])
    
            
        