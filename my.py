
import argparse
import os
from copy import deepcopy

import shutil
import time

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt

from config import (
    IMAGE_DECIMATION,
    IMAGE_SIZE, THROTTLE_BOUND, STEER_BOUND,
    BATCH_SIZE, NUM_EPOCHS,
    NUM_X_CHANNELS, NUM_X_DIFF_CHANNELS,
    TRAIN_SET, TEST_SET,
    WEIGHT_EXPONENT, MIN_SPEED,
    IMAGE_CLIP_UPPER, IMAGE_CLIP_LOWER,
    SPEED_AS_INPUT, OUTPUTS_SPEC,
    ERROR_PLOT_UPPER_BOUNDS, SCATTER_PLOT_BOUNDS,
    BASE_FONTSIZE
)


def extract_y(filename):
    relevant_component = filename.split('/')[1].split('_depth_data')[0]
    episode = filename.split('_depth_data')[1].split('.npy')[0]
    DF_log = pd.read_csv('logs/{}_log{}.txt'.format(relevant_component, episode))
    if 'speed' in DF_log:
        which_OK = (DF_log['speed'] > MIN_SPEED)
        speed = DF_log[which_OK]['speed']
    else:
        which_OK = DF_log.shape[0] * [True]
        speed = pd.Series(DF_log.shape[0] * [-1])
    steer = DF_log[which_OK]['steer']
    throttle = DF_log[which_OK]['throttle']
    cte = DF_log[which_OK]['cte']
    epsi = DF_log[which_OK]['epsi']
    psi=DF_log[which_OK]['psi']
    x= DF_log[which_OK]['x']
    y= DF_log[which_OK]['y']
    
    return which_OK, steer, throttle, speed, cte, epsi, psi, x, y


def _get_data_from_one_racetrack(filename):
    which_OK, steer, throttle, speed, cte, epsi, psi, x, y = extract_y(filename)
    X = pd.np.load(filename)[..., which_OK].transpose([2, 0, 1])
    if X.shape[1] != (IMAGE_CLIP_LOWER-IMAGE_CLIP_UPPER) // IMAGE_DECIMATION:
        X = X[:, IMAGE_CLIP_UPPER:IMAGE_CLIP_LOWER, :][:, ::IMAGE_DECIMATION, ::IMAGE_DECIMATION]
        #X = X[:, ::IMAGE_DECIMATION, ::IMAGE_DECIMATION]

    # Need to expand dimensions to be able to use convolutions
    X = np.expand_dims(X, 3)
    return X, {
        'steer': steer.values,
        'throttle': throttle.values,
        'speed': speed.values,
        'cte': cte.values,
        'epsi': epsi.values,
        'psi': psi.values,
        'x': x.values,
        'y': y.values
    }

def get_data(filenames):
    X_all = []
    labels_all = []
    racetrack_labels = []
    for filename in filenames:
        X, labels = _get_data_from_one_racetrack(filename)
        X_all.append(X)
        labels_all.append(labels)
        racetrack_index = int(filename.split('racetrack')[1].split('_')[0])
        racetrack_labels += len(labels)*[racetrack_index]
        print("loading:  "+filename)

    label_names = labels.keys()
    X_out = np.concatenate(X_all)
    labels_out = {
        label_name: np.concatenate([labels[label_name] for labels in labels_all])
        for label_name in label_names
    }
    labels_out['racetrack'] = pd.get_dummies(racetrack_labels).values
    return X_out, labels_out

#for a series of inputs(input array), input array shape: (number of samples, 60, 80)
def predict_array(model,input_array):
    input_array=input_array*(-1)
    input_array=np.expand_dims(input_array, 3)
    output=model.predict(input_array)
    psi=output[...,0]/10.0
    epsi=output[...,1]/100.0
    cte=output[...,2]/10.0
    x=output[...,3]*5.0
    y=output[...,4]*5.0
    return psi,epsi,cte,x,y

#for single input, input shape: (60, 80)
def predict_single(model,input):
    input=input*(-1)
    input=np.expand_dims(input, 2)
    output=model.predict(np.array([input]))
    psi=output[0][0]/10.0
    epsi=output[0][1]/100.0
    cte=output[0][2]/10.0
    x=output[0][3]*5.0
    y=output[0][4]*5.0
    return psi,epsi,cte,x,y

def main1():
    train_X,train_Y=get_data(TRAIN_SET)
    input=np.array([train_X[50].reshape(60,80)])
    input1=np.array([train_X[50].reshape(60,80)])[0]
    saved_model_path='C://Users//wo d pc//Desktop//thelocationi//For Siyuan//mycnnall_2.h5'
    new_model = tf.keras.models.load_model(saved_model_path)
    #new_model.summary()
    pre=predict_array(new_model,input)
    pre1=predict_single(new_model,input1)
    print(pre)
    print(pre1)

def main():
    train_X,train_Y=get_data(TRAIN_SET)
    test_X,test_Y=get_data(TEST_SET)
    print(train_X[50].reshape(60,80).shape)
    input()
    train_X=train_X*(-1)
    test_X=test_X*(-1)
    #print(test_X.shape)
    train_psi_label=np.array(train_Y['psi'])
    test_psi_label=np.array(test_Y['psi'])
    model = tf.keras.models.Sequential()
    input_shapemy = ((IMAGE_CLIP_LOWER-IMAGE_CLIP_UPPER) // IMAGE_DECIMATION,IMAGE_SIZE[1] // IMAGE_DECIMATION,NUM_X_CHANNELS+NUM_X_DIFF_CHANNELS)
    l2_reg=0.00001

    #0.1
    model.add(layers.Conv2D(filters=12*2, kernel_size=(7, 7), strides=(1, 1), activation='elu', padding='same', kernel_regularizer=l2(l2_reg), input_shape=input_shapemy))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    #1
    model.add(layers.Conv2D(filters=24*2, kernel_size=(5, 5), strides=(1, 1), activation='elu', padding='same', kernel_regularizer=l2(l2_reg)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    #2
    model.add(layers.Conv2D(filters=48*2, kernel_size=(5, 5), strides=(1, 1), activation='elu', padding='same', kernel_regularizer=l2(l2_reg)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    #3
    model.add(layers.Conv2D(filters=96*2, kernel_size=(5, 5), strides=(1, 1), activation='elu', padding='same', kernel_regularizer=l2(l2_reg)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    #-----------------

    #model.add(layers.Dropout(0.2))
    #Flatten the CNN output so that we can connect it with fully connected layers
    model.add(layers.Flatten())
    # FC6 Fully Connected Layer
    model.add(layers.Dense(512, kernel_regularizer=l2(l2_reg), activation='elu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, kernel_regularizer=l2(l2_reg), activation='elu'))
    model.add(layers.Dropout(0.2))

    #Output Layer with softmax activation
    model.add(layers.Dense(128, kernel_regularizer=l2(l2_reg), activation='elu'))
    model.add(layers.Dense(1, kernel_regularizer=l2(l2_reg), activation='linear'))

    optimizer = tf.keras.optimizers.Adam(0.0003) #0.0003
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    hist = model.fit(train_X, train_psi_label, epochs=100, batch_size=128, validation_split=0.1, verbose=2)
    model.summary()

    model.save('mycnn.h5')
    test_score = model.evaluate(test_X,  test_psi_label, verbose=2)
    print(test_score)

    #models.predict()


    return 0
   
if __name__ == '__main__':
    main1()


#input -> Convolutional -> fullyconnection -> ouput
#1复合结构
#input -> Convolutional -> fullyconnection -> mid -> fullyconnection' ->output1
#                                                 -> fullyconnection'' ->output2
#2并入分支结构
#input1 -> Convolutional -> fullyconnection -> mid -> fullyconnection ->output
#               input2 (-> fullyconnection) -> 