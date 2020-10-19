#this tutorial is written by jason, so some spelling mistakes should be excused
'''
it is always wise to create a validation set to make sure that our predictions are as
well predicted as we need them, if you train a set that is a lot different from the real
validation data, you may have to consider retraining them, or maybe something is wrong with
your programming

over-fiting is a really annyoing thing, it means that even though your data are trained well
for the trained set an be able to yield good results, it is too focused on it. it will give
less than good result for the validation set, then we need to take the specific steps
necessary to solve it
'''

#import modules
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels=[]
train_samples=[]

'''
example data:
* an experiemental drug was tested on person from 13 to 100 years old in a clinical trail
* the trial had 2100 participants. Half are under 65, other half is older
* around 95% of patients 65 or older experienced side effects
* around 95% of patients under 65 experienced no side effects
'''

for i in range(50):
    #the 5 percent of younger individuals who did experience side effects
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    #the 5 percent of older individuals who did not experience side effects
    random_younger=randint(65,100)
    train_samples.append(random_younger)
    train_labels.append(0)

for i in range(1000):
    #the 95 percent of younger individuals who did not experience side effects
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    #the 5 percent of older individuals who did experience side effects
    random_younger=randint(65,100)
    train_samples.append(random_younger)
    train_labels.append(1)
'''
for i in train_samples:
    print(i)
'''
#normalize and split the data
train_labels=np.array(train_labels)
train_samples=np.array(train_samples)
train_labels, train_samples=shuffle(train_labels, train_samples)##shuffle them, get rid of orders that may have been in the data

scaler=MinMaxScaler(feature_range=(0, 1))
scaled_train_samples=scaler.fit_transform(train_samples.reshape(-1,1))


##datas have been normalized

##import tensorflows
import tensorflow as td
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

#layers
model=Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
    ])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)
#notice i add a valudation_split to use keras to split some proportion of the training set
