import cv2
import os 
import numpy as np
import pandas as pd

from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

from keras_preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Concatenate, Conv3D, MaxPooling2D

baseKernels = []
baseKernels.append((1,4,6,4,1))
baseKernels.append((1,-2,0,-2,1))
baseKernels.append((-1,0,2,0,-1))
baseKernels.append((1,-4,6,-4,1))
baseKernels = np.array(baseKernels)

def comb_files(directory, labels_file):
    read_files = []
    paired_files = []
    dropped_files = []
    images = []
    counter = 0

    labels = pd.read_pickle(labels_file)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        read_files.append(filename)
        print("Saved File")


    for val in labels.index.values:
        if val in read_files:
            paired_files.append(val)
            counter += 1
        else:
            dropped_files.append(counter)
            counter += 1

    labels.drop(labels.index[dropped_files])
    print(labels.shape)

    paired_files = np.array(paired_files)

    read_files.clear()
    del read_files[:]

    dropped_files.clear()
    del dropped_files[:]

    for file in paired_files:
        images.append(np.array(cv2.imread(directory + "/" + file)))
        print("Opened Image ", file)

    images = np.array(images)

    return images, labels

#take images and process each image to make array of 16 convoluted versions of the base and add that to massive numpy
def kernelPreProcess(raw_images):
    processed_images = []
    for n in range(len(raw_images)):
        #pre process by applying 16 laws masks
        imageForms = []
        for x in range(4):
            for y in range(4):
                imageForms.append(cv2.filter2D(src=raw_images[n],ddepth=-1,kernel=np.multiply(baseKernels[x],np.reshape(baseKernels[y],[5,1]))))
                print("Image ",n," Filter ",x, " ",y)
        imageForms = np.array(imageForms)
        processed_images.append(imageForms)
        print("Processed Image ",n)
    processed_images = np.array(processed_images)
    return processed_images

def image_transform(image):
    imageForms = image
    #for x in range(4):
    #    for y in range(4):
    #        imageForms.append(cv2.filter2D(src=image,ddepth=-1,kernel=np.multiply(baseKernels[x],np.reshape(baseKernels[y],[5,1]))))
    #imageForms = sum(imageForms)
    return imageForms

#get directory of input images and create array of images and store images in the directory to the array
train_dir = "C:/Users/panka/Desktop/Aditya/image data 2018-19/Train_Resized"
#get labels pickle and convert to dataframe then sort by the filename to go along with the images
train_labels_file = "C:/Users/panka/Desktop/Aditya/image data 2018-19/Training_Input_Resized.pkl"

train_labels = pd.read_pickle(train_labels_file)

train_datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=image_transform)
train_generator = train_datagen.flow_from_dataframe(dataframe=train_labels,directory=train_dir,target_size=(108,192),x_col='Filename',y_col=['Right Ankle x','Right Knee x','Right Hip x','Left Hip x','Left Knee x','Left Ankle x','Pelvis x','Thorax x','Upper Neck x','Head Top x','Right Wrist x','Right Elbow x','Right Shoulder x','Left Shoulder x','Left Elbow x','Left Wrist x','Right Ankle y','Right Knee y','Right Hip y','Left Hip y','Left Knee y','Left Ankle y','Pelvis y','Thorax y','Upper Neck y','Head Top y','Right Wrist y','Right Elbow y','Right Shoulder y','Left Shoulder y','Left Elbow y','Left Wrist y'],class_mode='other',batch_size=8)
    
#get directory of input images and create array of images and store images in the directory to the array
test_dir = "C:/Users/panka/Desktop/Aditya/image data 2018-19/Test_Resized"
#get labels pickle and convert to dataframe then sort by the filename to go along with the images
test_labels_file = "C:/Users/panka/Desktop/Aditya/image data 2018-19/Testing_Input_Resized.pkl"

test_labels = pd.read_pickle(test_labels_file)

test_datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=image_transform)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_labels,directory=test_dir,target_size=(108,192),x_col='Filename',y_col=['Right Ankle x','Right Knee x','Right Hip x','Left Hip x','Left Knee x','Left Ankle x','Pelvis x','Thorax x','Upper Neck x','Head Top x','Right Wrist x','Right Elbow x','Right Shoulder x','Left Shoulder x','Left Elbow x','Left Wrist x','Right Ankle y','Right Knee y','Right Hip y','Left Hip y','Left Knee y','Left Ankle y','Pelvis y','Thorax y','Upper Neck y','Head Top y','Right Wrist y','Right Elbow y','Right Shoulder y','Left Shoulder y','Left Elbow y','Left Wrist y'],class_mode='other',batch_size=8)

#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, data_format="channels_last", kernel_size=3, input_shape=(108,192,3), activation='relu'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#train the model
STEP_SIZE_TRAIN = train_generator.n//8
STEP_SIZE_TEST = test_generator.n//8
model.fit_generator(train_generator,epochs=50,validation_data=test_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_steps=STEP_SIZE_TEST)
