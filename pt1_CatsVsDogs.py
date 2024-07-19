import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline
import matplotlib.pyplot as plt

# fix from: https://stackoverflow.com/questions/48225729/importerrorcould-not-import-pil-image-working-with-keras-ternsorflow/52230898#52230898
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.image import load_img

# https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
# too bad...I dont hgave Nvidia
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Organize data into train, valid, test dirs
os.chdir('data/dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')
    
    for c in random.sample(glob.glob('cat*'), 500):
        shutil.move(c,'train/cat')
    for c in random.sample(glob.glob('dog*'), 500):
        shutil.move(c,'train/dog')
    for c in random.sample(glob.glob('cat*'), 100):
        shutil.move(c,'valid/cat')
    for c in random.sample(glob.glob('dog*'), 100):
        shutil.move(c,'valid/dog')
    for c in random.sample(glob.glob('cat*'), 50):
        shutil.move(c,'test/cat')
    for c in random.sample(glob.glob('dog*'), 50):
        shutil.move(c,'test/dog')
    
os.chdir('../../')

train_path = 'data/dogs-vs-cats/train'
valid_path = 'data/dogs-vs-cats/valid'
test_path = 'data/dogs-vs-cats/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

assert train_batches.n == 1000
assert valid_batches.n == 1000
assert test_batches.n == 1000

imgs, labels = next(train_batches)

# This function wiLL plot images in the form of a grid with 1 row and 10 coLumns where images are plac
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    
plotImages(imgs)
print (labels)

# 12. (01:19:22) Build and Train a CNN with TensorFlow's Keras API
# https://deeplizard.com/learn/video/daovGOlMbT4

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches,
    validation_data=valid_batches,
    epochs=10,
    verbose=2
)

# model.fit(x=train_batches,
#     steps_per_epoch=len(train_batches), # not required in this TF version?
#     validation_data=valid_batches,
#     validation_steps=len(valid_batches),
#     epochs=10,
#     verbose=2
# )

