import random
import matplotlib
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, InputLayer, Flatten, Activation, Dropout
from keras.models import Sequential, Model
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

data_folder = Path("C:/Users/kbinw/Binwant/01 IGDTUW/7th Sem/Final YR Project/Code 4/Sign_lang_project/")

# show example train images
plt.figure(figsize=(20, 20))
_folder = data_folder / "Data/A"

for i in range(5):
    file = random.choice(os.listdir(_folder))
    image_path = os.path.join(_folder, file)
    img = mpimg.imread(image_path)
    ax = plt.subplot(1, 5, i+1)
    ax.title.set_text(file)
    plt.imshow(img)

IMG_WIDTH = 300
IMG_HEIGHT = 300
img_width, img_height = 300, 300
img_folder = data_folder/"Data"


def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name


# extract the image array and class name
img_data, class_name = create_dataset(img_folder)

target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
target_dict

with open(data_folder/"Models/label.txt", 'w') as f:
    for key, value in target_dict.items():
        f.write('%s %s\n' % (value, key))

target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
print(target_dict)

epochs = 10
batch_size = 16
input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x=np.array(img_data, np.float32), y=np.array(list(map(int,target_val)), np.float32), epochs=5)

model.save(data_folder/"Models/model_1.h5")


