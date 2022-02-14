from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import matplotlib.image as mping
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

base_dir='/Users/naoki/maskdetect/New Masks Dataset'
train_dir=os.path.join(base_dir,'Train')
test_dir=os.path.join(base_dir,'Test')
valid_dir=os.path.join(base_dir,'Validation')

train_mask_dir=os.path.join(train_dir,"Mask")
train_nomask_dir=os.path.join(train_dir,"Non Mask")

train_mask_files=os.listdir(train_mask_dir)
train_nomask_files=os.listdir(train_nomask_dir)
nrows=4
ncols=4
plt.figure(figsize=(12,12))

mask_images=[]
for i in train_mask_files[0:8]:
  mask_images.append(os.path.join(train_mask_dir,i))

nomask_images=[]
for i in train_nomask_files[0:8]:
  mask_images.append(os.path.join(train_nomask_dir,i))
print(mask_images)
print(nomask_images)

merged_images=mask_images+nomask_images

train_datagene=ImageDataGenerator(rescale=1./255,
                                  zoom_range=0.2,
                                  rotation_range=50,
                                  horizontal_flip=True)
test_datagene=ImageDataGenerator(rescale=1./255)
valid_datagene=ImageDataGenerator(rescale=1./255)

train_generarte=train_datagene.flow_from_directory(train_dir,
                                                   target_size=(150,150),
                                                   batch_size=32,
                                                   class_mode='binary')

test_generarte=test_datagene.flow_from_directory(test_dir,
                                                   target_size=(150,150),
                                                   batch_size=32,
                                                   class_mode='binary')

valid_generarte=valid_datagene.flow_from_directory(valid_dir,
                                                   target_size=(150,150),
                                                   batch_size=32,
                                                   class_mode='binary')






model=Sequential()
model.add(Conv2D(32,(3,3),padding='SAME',activation='relu',input_shape=(150,150,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(64,(3,3),padding='SAME',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])


history=model.fit(train_generarte,
                  epochs=30,
                  validation_data=valid_generarte)

model.summary()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Tranimg Loss/validatioiin Loss')
plt.xlabel('epochs')

model.save('maskmodel01.h5')
