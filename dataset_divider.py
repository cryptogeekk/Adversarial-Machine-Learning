from tensorflow import keras
import pandas as pd
import numpy as np

def divide(parts, X_train_full,y_train_full):
    # fashion_mnist = keras.datasets.mnist
    # (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    
    # # for temproary use
    # X_train_full=np.arange(1,21)
    # y_train_full=np.arange(21,41)
    # parts=5
    
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

    