import random
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.models import Sequential, Model
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data_folder = Path("/home/dic-lb/PycharmProjects/Sign_lang_project/")

img_width, img_height = 224, 224
img_folder = data_folder/"Dataset"

n_epoch = 1
batch_sz = 20
input_shape = (img_width, img_height, 3)

mobile = tf.keras.applications.mobilenet.MobileNet()

# imports the mobilenet model and discards the last 1000 neuron layer
base_model = MobileNet(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)

# add dense layers so that the model can learn more complex functions and classify for better results
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)

# final layer with softmax activation
preds = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

model.summary()

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(img_folder,
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=batch_sz,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory(img_folder,
                                                         target_size=(224, 224),
                                                         color_mode='rgb',
                                                         batch_size=batch_sz,
                                                         class_mode='categorical',
                                                         shuffle=True,
                                                         subset='validation')

#test_generator = ImageDataGenerator().flow_from_directory(img_folder/"test",
#                                                          shuffle=False,
#                                                          target_size=(224, 224),
#                                                          color_mode='rgb',
#                                                          batch_size=batch_sz)

step_size_train = train_generator.n//train_generator.batch_size
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples//validation_generator.batch_size,
                    steps_per_epoch=step_size_train,
                    epochs=n_epoch)

predictions = model.predict_generator(train_generator, test_generator.samples//batch_sz+1)
pred = np.argmax(predictions, axis=1)
cm = confusion_matrix(train_generator.classes, pred)

print('Confusion Matrix')
print(cm)
print('Classification Report')
target_names = ['0', '1', '2', '3', '4']
print(classification_report(train_generator.classes, pred, target_names=target_names))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Save the trained model for testing and classification in real-time
model.save(data_folder/"Models/MobileNetModel.h5")
