#! /usr/bin/env python

import numpy as np
import scipy
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Dense, LSTM, InputLayer
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import regularizers


def gen_batch_ae(X_train, X_val, batch_size = 32, shuffle_buffer = 1000):
    """
    Utility function to create a minibatch generator
    for Autoencoder training
    using tensorflow.data.dataset module
    """
    X_train = tf.convert_to_tensor(X_train,dtype=tf.float64)
    X_val = tf.convert_to_tensor(X_val,dtype=tf.float64)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, X_val))
  
    train_dataset = train_dataset.batch(batch_size).shuffle(shuffle_buffer)
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset