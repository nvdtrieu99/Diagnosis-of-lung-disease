
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.sequential import relax_input_shape

# dựng model
model = Sequential()

#với mỗi layer conv kèm theo 1 layer pooling để thu nhỏ size
model.add(Conv2D(64,kernel_size=(3,3),activation='relu', input_shape=(28,28,1),padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Flatten())
#softmax phù hợp với bào toán có nhiều hơn 2 output
model.add(Dense(10,activation='softmax'))

#Compile và kiểm tra model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


