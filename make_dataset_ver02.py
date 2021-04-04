
import tensorflow
from tensorflow.keras.models import * 
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


#############
img_height, img_width = 224,224
train_data_dir = 'train/'
validation_data_dir ='validation/'
test_data_dir = 'test/'
batch_size = 3

######
#Data Agumentation
datagen = ImageDataGenerator(
    # rescale= 1. /255,
    shear_range= 0.2,
    zoom_range= 0.2,
    horizontal_flip= True,
    rotation_range= 20,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    validation_split= 0.5
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size= batch_size,
    subset= "training",
    class_mode= "categorical"
)

test_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width,img_height),
    batch_size= batch_size,
    # subset= "testing",
    class_mode= "categorical"
)
val_datagen = ImageDataGenerator()

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_height),
    batch_size= batch_size,
    subset= "validation",
    class_mode= "categorical"
)

# dựng model
model = Sequential()

#với mỗi layer conv kèm theo 1 layer pooling để thu nhỏ size
model.add(Conv2D(64,kernel_size=(3,3),activation='relu', input_shape=(img_height,img_width,3),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Flatten())
#softmax phù hợp với bào toán có nhiều hơn 2 output
model.add(Dense(3,activation='softmax'))

#Compile và kiểm tra model

# model.summary()
##
# rms = RMSprop(learning_rate=0.05,rho=0.9)
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['categorical_accuracy']
)

##
model.fit_generator(
    train_generator,
    steps_per_epoch=5,
    epochs=10,
    validation_data=validation_generator
)
model.save('model_h5/model_custome_ver02.h5')

# pred = model.predict(test_generator)

# print(pred)

