# Adapted from https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# Get all paths, shuffle paths, create dataset from paths, scale 
data, labels = [], []

imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten() # resize all images to same dims regardless of aspect ratio, 
    data.append(image)

    label = imagePath.split(os.path.sep)[-2] # extract class label from image path
    labels.append(label)

data = np.array(data, dtype="float") / 255.0 # scale to 0-1
labels = np.array(labels)

# Sort into train/test batches
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# Turn labels into vector representing class i.e. dog, cat, panda
# dog => [1, 0, 0]
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Create, compile, fit model
simple = Sequential()
simple.add(Dense(1024, input_shape=(3072,), activation="sigmoid")) # 1072 corresponds to 32 * 32 * 3 pixels represented as flattened arrays
simple.add(Dense(512, activation="sigmoid"))
simple.add(Dense(len(lb.classes_), activation="softmax")) # final layer classifies to num classes i.e. 3, softmax activation

# compile model
INITIAL_LR = 0.01
EPOCHS = 80

opt = SGD(learning_rate=INITIAL_LR)
simple.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy']) 
# Always use categorical cross-entropy unless binary classification, use binary cross-entropy

H = simple.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)
# batch size = # images pass through network while running GD
# Depends on GPU

# Evaluate model
print("[INFO] evaluating network...")
predictions = simple.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

print("[INFO] serializing network and label binarizer...")
simple.save(args["model"], save_format="h5")
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
