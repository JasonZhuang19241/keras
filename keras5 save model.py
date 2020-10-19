#this tutorial is written by jason, so some spelling mistakes should be excused
'''
predict data from test sets 42:31
'''

#import modules
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as td
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
model=Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
    ])

test_labels=[]
test_samples=[]

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
    test_samples.append(random_younger)
    test_labels.append(1)

    #the 5 percent of older individuals who did not experience side effects
    random_younger=randint(65,100)
    test_samples.append(random_younger)
    test_labels.append(0)

for i in range(1000):
    #the 95 percent of younger individuals who did not experience side effects
    random_younger=randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    #the 5 percent of older individuals who did experience side effects
    random_younger=randint(65,100)
    test_samples.append(random_younger)
    test_labels.append(1)

#normalize and split the data
test_labels=np.array(test_labels)
test_samples=np.array(test_samples)
test_labels, test_samples=shuffle(test_labels, test_samples)##shuffle them, get rid of orders that may have been in the data

scaler=MinMaxScaler(feature_range=(0, 1))
scaled_test_samples=scaler.fit_transform(test_samples.reshape(-1,1))


#predict!!!!!!! no output used so use no verbose in these 0 means no side effect, 1 means yes
predictions=model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

rounded_predictions=np.argmax(predictions, axis=-1)

''' confusion matrix
'''

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm=confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
cmap=plt.cm.get_cmap("Spectral")

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

cm_plot_labels=['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

plt.show()

#save model
import os.path
try:
    os.mkdir('models')
except:
    odajn=1
if os.path.isfile('models/medical_trial_model.h5') is False:
    model.save('models/medical_trial_model.h5')
