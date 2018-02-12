# Importing all required libraries
import numpy as np
import keras
import tensorflow as tf
from multiprocessing import Pool
from keras.constraints import maxnorm
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils
K.set_image_dim_ordering('th')
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import json
import threading
import h5py
from sklearn.utils import class_weight

# With GPU
with tf.device('/gpu:0'):

	# Threadlocking for Generator
    class threadsafe_iter:
        def __init__(self, it):
            self.it = it
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def next(self):
            with self.lock:
                return self.it.next()

    # Threadsafe for Training Generator
    def threadsafe_generator_train(f):
        def g(*a, **kw):
            return threadsafe_iter(f(*a, **kw))
        return g

    # Dataset location
    h5_file_location = '/home/chandu/Desktop/main_dataset_ocr/final_dataset.hdf5'
    dataset = h5py.File(h5_file_location, 'r')

    # Loading Train
    X_train = dataset[u'X_train']
    Y_train = np_utils.to_categorical(dataset[u'Y_c_train'])

    # Loading Validation
    X_val = dataset[u'X_val']
    Y_val = np_utils.to_categorical(dataset[u'Y_c_val'])
    
    # Loading Test
    X_test = dataset[u'X_test']
    Y_test = np_utils.to_categorical(dataset[u'Y_c_test'])

    # Class Rebalancing for skewed classes
    class_weight = class_weight.compute_class_weight('balanced', np.unique(dataset[u'Y_c_train']), dataset[u'Y_c_train'])

    # Generating a Random permutation for Training 
    training_random_indexes = np.random.permutation(len(Y_train))

    # Batch Size for Training
    batch_size_train = 500

    # Number of classes
    num_classes = Y_test.shape[1]

    # Epochs for training
    epochs = 15

    # Maximum number of iterations for each epoch
    max_train_iter_epoch = np.ceil( float(len(Y_train)) / float(batch_size_train))

    # Threadsafe Generator
    @threadsafe_generator_train
    def train_generator():
        while 1:
            for count_train in xrange(int(max_train_iter_epoch)):
                if count_train<int(max_train_iter_epoch)-1:
                    x_train = X_train[np.sort(training_random_indexes[count_train*batch_size_train:(count_train+1)*batch_size_train]),:]
                    y_train = Y_train[np.sort(training_random_indexes[count_train*batch_size_train:(count_train+1)*batch_size_train]),:]
                else :
                    x_train = X_train[len(Y_train)-batch_size_train:len(Y_train),:]
                    y_train = Y_train[len(Y_train)-batch_size_train:len(Y_train),:]
                x_train = np.divide(np.asarray(x_train,dtype='float32'),255.0)
                yield (x_train,y_train)

    # Create a Model
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(20,(3,3),input_shape=(1,32,32),padding='same',activation='relu',W_constraint=maxnorm(3)))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.convolutional.Conv2D(50,(3,3),activation='relu',padding='same',W_constraint=maxnorm(3)))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.convolutional.Conv2D(100,(3,3),activation='relu',padding='same',W_constraint=maxnorm(3)))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(500,activation='relu',W_constraint=maxnorm(3)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(num_classes,activation='softmax'))
    
    # Load Model if wanted to resume training
    # model.load_weights("resume_chars_weights.hdf5")

    # Write to JSON Model
    json_txt = model.to_json()
    with open('model_code_TCCNN-L.json','w') as outfile:
      json.dump(json_txt,outfile)
    outfile.close()

    # Compile Model
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    print model.summary()

    # Output Weights 
    filepath="model_code_TCCNN-L.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit model
    model.fit_generator( train_generator(), validation_data=(np.asarray(X_val,dtype=np.float32)/255.0, Y_val), 
                        verbose=1, callbacks=callbacks_list,
                        steps_per_epoch=int(max_train_iter_epoch),
                        nb_epoch=epochs, 
                        nb_worker=4, use_multiprocessing=False,
                        class_weight=class_weight )
    del X_val
    del Y_val
    del X_train
    del Y_train

    # Evaluate on Test Set
    scores = model.evaluate(np.asarray(X_test,dtype=np.float32)/255.0, Y_test, verbose=1)
    
    print '\n On Test :::  Loss : ' + str(scores[0]) + '   -    Accuracy  :  ' + str(scores[1]) 
