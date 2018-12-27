import cv2
import os 
import numpy as np
import pandas as pd

from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

from keras.datasets import mnist
from keras.utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

baseKernels = []
baseKernels.append(np.array(1,4,6,4,1))
baseKernels.append(np.array(1,-2,0,-2,1))
baseKernels.append(np.array(-1,0,2,0,-1))
baseKernels.append(np.array(1,-4,6,-4,1))

def kernelPreProcess(raw_images):
    processed_images = np.zeros((raw_images.shape, 16))
    for n in range(raw_images):
        #pre process by applying 16 laws masks
        imageForms = []
        for x in range(4):
            for y in range(4):
                imageForms.append(cv2.filter2D(src=raw_images[n],ddepth=-1,kernel=np.multiply(baseKernels[x],np.reshape(baseKernels[y],[5,1]))))
        processed_images.append(imageForms)

#get directory of input images
train_dir = askdirectory()
#get labels pickle and convert to dataframe
labels_file = askopenfilename()
labels = pd.read_pickle(labels_file)

#create array of images and store images in the directory to the array
train_images = []
for file in os.listdir(train_dir):
    filename = os.fsdecode(file)
    img = cv2.imread(dir + "/" + filename, 1)
    images.append(img)

#get filename of .csv file of the image labels
train_filename = askopenfilename()
train_labels = np.genfromtxt(filename, delimiter=",", skip_header=1)

#images are x and labels are y
X_train = kernelPreProcess(train_images)
y_train = kernelPreProcess(train_labels)

#get directory of input images
test_dir = askdirectory()

#create array of images and store images in the directory to the array
test_images = []
for file in os.listdir(test_dir):
    filename = os.fsdecode(file)
    img = cv2.imread(dir + "/" + filename, 1)
    images.append(img)

#get filename of .csv file of the image labels
test_filename = askopenfilename()
test_labels = np.genfromtxt(filename, delimiter=",", skip_header=1)

#images are x and labels are y
X_test = kernelPreProcess(test_images)
y_test = kernelPreProcess(test_labels)

#download mnist data and split into train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshape data to fit model
#X_train = X_train.reshape(60000,28,28,1)
#X_test = X_test.reshape(10000,28,28,1)

#one-hot encode target column
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predict first 4 images in the test set
model.predict(X_test[:4])

#actual results for first 4 images in test set
print(y_test[:4])
