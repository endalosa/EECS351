import numpy as np
from time import sleep
from tqdm import tqdm
from numpy import asarray
import cv2 

from PIL import Image, ImageEnhance

from os.path import dirname, join as pjoin
import scipy.io as sio

from sklearn.neighbors import KNeighborsClassifier
from IPython.display import clear_output

import timeit

from sklearn.metrics import confusion_matrix
from sklearn import tree

import matplotlib.pyplot as plt

mat = sio.loadmat('80_20_data.mat')

# Simplify variable names
train_cat = mat['TRAIN_CAT']
train_dog = mat['TRAIN_DOG']
test_cat = mat['TEST_CAT_NEW']
test_dog = mat['TEST_DOG_NEW']

# Make label vectors for train
labels = ["cat"]
for i in range(1,10000):
    labels = np.concatenate((labels, ["cat"]), axis=0)
for i in range(0,10000):
    labels = np.concatenate((labels, ["dog"]), axis=0)

# Turn images black and white
black_and_white_train = []
for i in range(0,10000):
    image = cv2.cvtColor(train_cat[:,:,:,i], cv2.COLOR_RGB2GRAY)
    black_and_white_train.append(image.flatten())
for i in range(0,10000):
    image = cv2.cvtColor(train_dog[:,:,:,i], cv2.COLOR_RGB2GRAY)
    black_and_white_train.append(image.flatten())
    
    # Make label vector for test
CorrectLabels = ["cat"]
for i in range(1,2500):
    CorrectLabels = np.concatenate((CorrectLabels, ["cat"]), axis=0)
for i in range(0,2500):
    CorrectLabels = np.concatenate((CorrectLabels, ["dog"]), axis=0)
    
# Turn images black and white
black_and_white_test = []
for i in range(0,2500):
    image = cv2.cvtColor(test_cat[:,:,:,i], cv2.COLOR_RGB2GRAY)
    black_and_white_test.append(image.flatten())
for i in range(0,2500):
    image = cv2.cvtColor(test_dog[:,:,:,i], cv2.COLOR_RGB2GRAY)
    black_and_white_test.append(image.flatten())
    
#Decision Tree Code
# Create Model
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(black_and_white_train, labels)
#predicted_groups = clf.predict(black_and_white_test)
# cm = confusion_matrix(CorrectLabels, predicted_groups)
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
# disp.plot()
# plt.show()

#KNN 
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(black_and_white_train, labels)
predicted_groups = neigh.predict(black_and_white_test)
cm = confusion_matrix(CorrectLabels, predicted_groups)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=neigh.classes_)
disp.plot()
plt.show()
