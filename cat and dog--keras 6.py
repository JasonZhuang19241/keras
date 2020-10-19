#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
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


print("all modules imported")

#Data preparation into train, valid, test dirs
os.chdir('D:/OneDrive/桌面/python/dogs_vs_cats/train')
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

train_path='D:/OneDrive/桌面/python/dogs_vs_cats/train/train'
valid_path='D:/OneDrive/桌面/python/dogs_vs_cats/train/valid'
test_path='D:/OneDrive/桌面/python/dogs_vs_cats/train/test'
#process
train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['cat','dog'], batch_size=10)
valid_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['cat','dog'], batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['cat','dog'], batch_size=10, shuffle=False)


imgs, labels=next(train_batches)

#this function will make a image in the form od a grid with 1 rows and 10 columns where the images are placed

def plotImages(images_arr):
    fig, axes=plt.subplots(1,10, figsize=(20,20))
    axes=axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)
print(labels)

###build and train a CNN
model=Sequential([
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2,2), strides=2),
    Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2,2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
    ])
print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=10)


test_imgs, test_labels=next(test_batches)
plotImages(test_imgs)
print(test_labels)

test_batches.classes


predictions=model.predict(x=test_batches, verbose=0)#no output
print(np.round(predictions))



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion matrix"):
    """
this functio prints and plots the confusion matrix.
normalization can be applied by setting `normalize=True`.
"""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm= cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusiong matrix, without normalization')
    print(cm)

    thresh=cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




cm=confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))




from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


cmap=plt.cm.get_cmap("Spectral")

test_batches.class_indices

cm_plot_labels=['cat', 'dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


#definitely over fitting, but move on now

