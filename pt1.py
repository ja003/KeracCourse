import numpy as np
from random import randint 
from sklearn.utils import shuffle 
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

train_labels = []
train_samples = []

for i in range(50):
    random_youger = randint(13, 64)
    train_samples.append(random_youger)
    train_labels.append(1)
    
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)
    
for i in range(1000):
    random_youger = randint(13, 64)
    train_samples.append(random_youger)
    train_labels.append(0)
    
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)
    
for i in train_samples:
    print(i)
    
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_sample = scaler.fit_transform(train_samples.reshape(-1,1))

for i in scaled_train_sample:
    print(i)

# enable GPU
# physical_devices = tf.config.experimental. 'GPU' )
# print("Num GPUs Available:", len (physical_devices) )
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# model = Sequential([
#     Dense(units=16, input_shape=(1,), activation="relu"),
#     Dense(units=32, activation="relu"),
#     Dense(units=2, activation="softmax")
# ])
# newer version: use Input as separate layer
model = Sequential([
    Input(shape=(1,)),
    Dense(units=16, activation="relu"),
    Dense(units=32, activation="relu"),
    Dense(units=2, activation="softmax")
])

print(model.summary())