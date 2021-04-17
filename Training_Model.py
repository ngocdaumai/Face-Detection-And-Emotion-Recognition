import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

path = "SMILEs"
myList = os.listdir(path)
images = []
classNo = []
testRatio = 0.2
valRatio = 0.2
imageDimensions= (32, 32, 1)
print((myList))
noOfCLasses = len(myList)
# print(noOfCLasses)

print("Importing Classes........")
classes = myList
for x in classes:
    myPicList = os.listdir(path+"/"+str(x))
    print(myPicList)
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg, (imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        if(x=="negatives"):classNo.append(0)
        else:classNo.append(1)
    print(x, end=" ")
print("Done!")
images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
# print(X_train.shape)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)
# print(X_train.shape)

# print(np.where(y_train==0))

numOfSamples = []
for x in range(0, noOfCLasses):
    numOfSamples.append(len(np.where(y_train==x)[0]))
print(numOfSamples)
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/225
    return img

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
y_train = to_categorical(y_train, noOfCLasses)
y_test = to_categorical(y_test, noOfCLasses)
y_validation = to_categorical(y_validation, noOfCLasses)
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNodes = 500
    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                               imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfCLasses, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model  = myModel()
print(model.summary())

batchSizeVal= 50
epochsVal = 10
stepsPerEpochVal = 300

history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batchSizeVal),
                    steps_per_epoch=stepsPerEpochVal,
                    epochs=epochsVal,
                    validation_data=(X_validation,y_validation),
                    shuffle=1)

#### PLOT THE RESULTS
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

#### EVALUATE USING TEST IMAGES
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

#### SAVE THE TRAINED MODEL
pickle_out= open("emotion_model.p", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()