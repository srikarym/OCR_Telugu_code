import numpy as np
import os
from PIL import Image
import cv2
import random
import glob
import keras
import tensorflow as tf
from multiprocessing import Pool
from keras.constraints import maxnorm
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils
K.set_image_dim_ordering('th')
# Some part
from collections import Counter
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import json
import time
import logging
import threading
import h5py
from sklearn.utils import class_weight

with tf.device('/gpu:0'):

    class threadsafe_iter:
        def __init__(self, it):
            self.it = it
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def next(self):
            with self.lock:
                return self.it.next()


    def threadsafe_generator_train(f):
        def g(*a, **kw):
            return threadsafe_iter(f(*a, **kw))
        return g

    h5_file_location = '../../final_dataset.hdf5'
    dataset = h5py.File(h5_file_location, 'r')

    X_train = dataset[u'X_train']
    Y_train = np.asarray(dataset[u'Y_v_g_train'],dtype='int')
    train_indexes = np.where(Y_train[:]!=-1)[0]

    X_val = dataset[u'X_val']
    Y_val = np.asarray(dataset[u'Y_v_g_val'],dtype='int')
    val_indexes = np.where(Y_val[:]!=-1)[0]
    X_val = np.divide(np.asarray(X_val[val_indexes,:],dtype=np.float32),255.0)
    Y_val = np_utils.to_categorical(Y_val[val_indexes])

    class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    print val_indexes
    print X_train.shape
    print Y_train.shape
    print X_val.shape
    print Y_val.shape

    size_image = [32,32]
    training_random_indexes = np.random.permutation(train_indexes)

    batch_size_train = 500

    max_train_iter_epoch = np.ceil( float(len(training_random_indexes)) / float(batch_size_train))

    @threadsafe_generator_train
    def train_generator():
        while 1:
            for count_train in xrange(int(max_train_iter_epoch)):
                if count_train<int(max_train_iter_epoch)-1:
                    x_train = X_train[np.sort(training_random_indexes[count_train*batch_size_train:(count_train+1)*batch_size_train]),:]
                    y_train = Y_train[np.sort(training_random_indexes[count_train*batch_size_train:(count_train+1)*batch_size_train])]
                else :
                    x_train = X_train[np.sort(training_random_indexes[len(training_random_indexes)-batch_size_train:len(training_random_indexes)]),:]
                    y_train = Y_train[np.sort(training_random_indexes[len(training_random_indexes)-batch_size_train:len(training_random_indexes)])]
                x_train = np.divide(np.asarray(x_train,dtype='float32'),255.0)
                yield (x_train,np_utils.to_categorical(y_train,num_classes))

    num_classes = Y_val.shape[1]
    

    # Create a Model
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(25,(3,3),input_shape=(1,32,32),padding='same',activation='relu',W_constraint=maxnorm(3)))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.convolutional.Conv2D(20,(3,3),activation='relu',padding='same',W_constraint=maxnorm(3)))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256,activation='relu',W_constraint=maxnorm(3)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(num_classes,activation='softmax'))

    json_txt = model.to_json()
    with open('model_code_TVCNN-S.json','w') as outfile:
      json.dump(json_txt,outfile)
    outfile.close()

    # Compile Model
    epochs = 40
    lrate = 0.01
    decay = lrate/epochs
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    print model.summary()

    filepath="model_code_TVCNN-S.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit model
    model.fit_generator( train_generator(), validation_data=(X_val, Y_val), 
                        verbose=1, callbacks=callbacks_list,
                        steps_per_epoch=int(max_train_iter_epoch),
                        nb_epoch=epochs, #validation_steps=int(max_val_iter_epoch),
                        nb_worker=1, use_multiprocessing=False,
                        class_weight=class_weight )
    del X_val
    del Y_val
    del X_train
    del Y_train
    
    X_test = dataset[u'X_test']
    Y_test = np.asarray(dataset[u'Y_v_g_test'],dtype='int')
    test_indexes = np.where(Y_test[:]!=-1)[0]
    Y_test = np_utils.to_categorical(Y_test[test_indexes])
    X_test = np.divide(np.asarray(X_test[test_indexes,:],dtype=np.float32),255.0)

    scores = model.evaluate(X_test, Y_test, verbose=1)
    
    print 'On Test :::  Loss : ' + str(scores[0]) + '   -    Accuracy  :  ' + str(scores[1]) 
