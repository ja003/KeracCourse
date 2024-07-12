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
    
# (00:47:48) Create a Confusion Matrix for Neural Network Predictions

#%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title= 'Confusion matrix' ,
                          cmap=plt.cm.Blues) :
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting=True
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))    
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=l)[:, np.newaxis]
        print("Norma1ized confusion matrix")
    else: 
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# (00:52:29) Save and Load a Model with TensorFlow's Keras API

#1. ??
model.summary()

import os.path
filename = 'models/medical_trial_model.h5' # legacy format
filename = 'models/medical_trial_model.keras' # newer format
if os.path.isfile(filename) is False:
    model.save(filename)
    
from tensorflow.keras.models import load_model
new_model = load_model(filename)
new_model.summary() # check that it is tha same model
new_model.get_weights()

# 2. model.to_json()

# save as JSON
json_string = model.to_json()
# save as YAML
# yaml_string = model.to_yaml()
json_string
# model reconstruction from JSON:
from tensorflow.keras.models import model_from_json
model_architecture = model_from_json(json_string)
model_architecture.summary()
# model reconstruction from YAML
# from tensorfLow. keras. modeLs import modeL_from_yamL
# model = modeL_from_yamL(yamL_string)

# 3. model.save_weights

# Checks first to see if fiLe exists aLready.
# If not, the weights are saved to disk.
import os.path
filename2 = 'models/my_model_weights.weights.h5'
if os.path.isfile(filename2) is False:
    model.save_weights(filename2)
    
model2 = Sequential([
    Input(shape=(1,)),
    Dense(units=16, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model2.load_weights(filename2)
model2.get_weights()