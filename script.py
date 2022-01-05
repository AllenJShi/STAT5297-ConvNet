# import packages
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten




############################# Global variables and environment setup #############################
try:
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
except:
    pass

# random seed
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)


# PLEASE BE SURE TO UPDATE THE PATH VARIABLE IF YOU STORE YOUR DATASETS SOMEWHERE ELSE
path = './dataset/SVHN/format2/{}_32x32.mat'

# Global parameters
input_shape = (32,32,3)
num_classes = 10


# define a helper function that loads datasets
def load(train=True, test=False):
    if train + test == 0:
        return
    if train and not test:
        token = ['train']
    elif test and not train:
        token = ['test']
    else:
        token = ['train','test']
        return list(map(lambda x: loadmat(file_name=path.format(x)),token))


################################# Data Preprocessing ################################
# load training and testing datasets
[train,test] = load(train=True,test=True)

# split inputs and labels
X_train, y_train = train['X'], train['y']
X_test, y_test = test['X'], test['y']


# Scale images to the [0, 1] range
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# transpose input matrix 
X_train = np.transpose(X_train,(3,0,1,2))
X_test = np.transpose(X_test,(3,0,1,2))



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train.astype('float32'))
y_test = keras.utils.to_categorical(y_test.astype('float32'))

# delete class 0
y_train = np.delete(y_train,(0),axis=1)
y_test = np.delete(y_test,(0),axis=1)



################################ Model Construction ####################################
# define the build model function that contains all models from the starting to the final model

def build_model(input_shape,num_classes,learning_rate,model='final'):

    keras.backend.clear_session() 
  
    if model=='starting':
############################# Starting Model ##################################
        model = Sequential(
            [
            Input(shape=input_shape),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
            ]
            )
###############################################################################
    elif model=='second':
############################# Second Model ##################################
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))
###############################################################################
    elif model=='third':
############################# Third Model ##################################
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
###############################################################################
    elif model=='final':
############################# Final Model ##################################
        model = Sequential([
          Conv2D(32, (3, 3), padding='same',activation='relu',input_shape=input_shape),
          BatchNormalization(),
          Conv2D(32, (3, 3), padding='same',activation='relu'),
          MaxPooling2D((2, 2)),
          Dropout(0.3),
          Conv2D(64, (3, 3), padding='same',activation='relu'),
          BatchNormalization(),
          Conv2D(64, (3, 3), padding='same',activation='relu'),
          MaxPooling2D((2, 2)),
          Dropout(0.3),
          Conv2D(128, (3, 3), padding='same',activation='relu'),
          BatchNormalization(),
          Conv2D(128, (3, 3), padding='same',activation='relu'),
          MaxPooling2D((2, 2)),
          Dropout(0.3),
          Flatten(),
          Dense(128, activation='relu'),
          Dropout(0.4),    
          Dense(num_classes,  activation='softmax')
          ])
###############################################################################

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
      loss='categorical_crossentropy',
      metrics=['accuracy'])
    model.summary()
    return model



# Hyperparameters
batch_size = 128
learning_rate = 1e-3
# Use for Stage I and II
# epochs = 5

# Use for Stage III
epochs = 10

validation_split = 0.1

early_stopping = EarlyStopping(
    monitor='loss',
    min_delta=0.0005,
    patience=3,
    mode='min'
    )


# Stage III: Data Augmentation
data_generator = ImageDataGenerator(
    rotation_range=9,
    height_shift_range=0.1,
    width_shift_range=0.1,
    vertical_flip=False,
    horizontal_flip=False
    )



##################################### Model Training ###################################
# with data augmentation
model_aug = build_model(input_shape,num_classes,learning_rate,'final')

history_aug = model_aug.fit_generator(
    data_generator.flow(X_train,y_train,batch_size),
    epochs=epochs,
    callbacks=[early_stopping]
    )

# without data augmentation
# model = build_model(input_shape,num_classes,learning_rate,'final')
# history = model.fit_generator(
#     ImageDataGenerator().flow(X_train,y_train,batch_size),
#     epochs=epochs,
#     callbacks=[early_stopping]
#     )


######################################### Model Evaluation ################################
loss_, acc_ = model_aug.evaluate(X_test,y_test,batch_size)
print(f'test loss (data augmentation): {loss_:0.2f}')
print(f'test accuracy (data augmentation): {acc_:0.2f}')

# loss, acc = model.evaluate(X_test,y_test,batch_size)
# print(f'test loss: {loss:0.2f}')
# print(f'test accuracy: {acc:0.2f}')