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
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)
    
for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
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
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

for i in scaled_train_samples:
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

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, shuffle=True, verbose=2)

# end of tutorial

# MY TRY - BEGIN

model.predict(scaled_train_samples)

#try predict unlabeled data

unlabeled_samples = []

for i in range(50):
    random_younger = randint(13, 64)
    unlabeled_samples.append(random_younger)
    
    random_older = randint(65, 100)
    unlabeled_samples.append(random_older)
    
unlabeled_samples = np.array(unlabeled_samples)
unlabeled_samples = shuffle(unlabeled_samples)
scaled_unlabeled_samples = scaler.fit_transform(unlabeled_samples.reshape(-1,1))
    
model.predict(scaled_unlabeled_samples)

# MY TRY - END

# (00:30:07) Build a Validation Set With TensorFlow's Keras API

# now we set validation_split 
# - splits portion of training set into validation set
# - split happens before shuffle
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)

# (00:39:28) Neural Network Predictions with TensorFlow's Keras API

test_labels = []
test_samples = []
scaler = MinMaxScaler(feature_range=(0,1))

# generate random data
for i in range(10):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)
    
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)
    
for i in range(200):
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)
    
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)
    
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

# predict

predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)
for i in predictions:
    print(i)
    
rounded_predictions = np.argmax(predictions, axis=-1)  
for i in rounded_predictions:
    print(i)
    
# print format: "age: will_have_side_effect"
for i in range(np.size(rounded_predictions)):
    print(str(test_samples[i]) + ": " + str(rounded_predictions[i]))
    
    