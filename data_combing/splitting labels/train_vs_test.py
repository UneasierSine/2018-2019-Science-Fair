import pandas as pd
import numpy as np
import csv
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfile

#read dataset labels
initial_file = askopenfilename()
dataset = pd.read_pickle(initial_file)

#get size of training and test dataset sizes
train_size = dataset.shape[0] * 0.8
print("Original Dataset Size: ",dataset.shape)

#arrays of the labels and then DataFrame for easy appending
training_rows = []
testing_rows = []

training_set = pd.DataFrame()
testing_set = pd.DataFrame()

#sort by filename
dataset.sort_index(inplace=True)
print(dataset)

#assign to training and testing label
for x in range(dataset.shape[0]):
    if x < train_size:
        training_rows.append(dataset.iloc[[x]])
    else:
        testing_rows.append(dataset.iloc[[x]])

#set the datasets of training and testing
training_set = pd.concat(training_rows)
testing_set = pd.concat(testing_rows)

print(dataset)

print("Training Dataset Size: ",training_set.shape)
print(training_set)

#save training set
#training_set.to_pickle("C:/Users/panka/Desktop/Aditya/image data 2018-19/Training.pkl")

print("Testing Dataset Size: ",testing_set.shape)
print(testing_set)

#save testing set
#testing_set.to_pickle("C:/Users/panka/Desktop/Aditya/image data 2018-19/Testing.pkl")

print(training_set[training_set.duplicated(keep=False)])
print(testing_set[testing_set.duplicated(keep=False)])
