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

data_folder = Path("C:/Users/kbinw/Binwant/01 IGDTUW/7th Sem/Final YR Project/Code 4/Sign_lang_project/")

img_width, img_height = 224, 224
img_folder = data_folder/"Data"


# def create_dataset(img_folder):
#     img_data_array = []
#     class_name = []
#
#     for dir1 in os.listdir(img_folder):
#         for file in os.listdir(os.path.join(img_folder, dir1)):
#             image_path = os.path.join(img_folder, dir1, file)
#             image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
#             image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
#             image = np.array(image)
#             image = image.astype('float32')
#             image /= 255
#             img_data_array.append(image)
#             class_name.append(dir1)
#     return img_data_array, class_name
#
#
# # extract the image array and class name
# img_data, class_name = create_dataset(img_folder)
# print(img_data)
#
# target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
# target_dict
#
# with open(data_folder/"Models/label.txt", 'w') as f:
#     for key, value in target_dict.items():
#         f.write('%s %s\n' % (value, key))
#
# target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
# print(target_dict)

epochs = 10
batch_size = 16
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
preds = Dense(23, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

model.summary()

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(data_folder / "Data/",
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=128,
                                                    class_mode='categorical',
                                                    shuffle=True)

step_size_train = train_generator.n//train_generator.batch_size
history = model.fit(train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=1)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Save the trained model for testing and classification in real-time
model.save(data_folder/"Models/MobileNetModel.h5")
