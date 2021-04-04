import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('model_h5/model_custome_ver02.h5')
dir_path = 'test/COVID19/' 
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

for i in os.listdir(dir_path):
    print(i)
    # Replace this with the path to your image
    image = Image.open(dir_path+i)
    # img = cv2.imread(dir_path+i)
    # if img.shape != (224,224,3):pass 

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    # image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    try:
        data[0] = normalized_image_array
    except:
        pass
    

    # run the inference
    prediction = model.predict(data)

    # print(prediction)
    covid = prediction[0,0]
    normal = prediction[0][1]
    pneumonia = prediction[0][2]

    max1 = max(covid,normal,pneumonia)

    if max1 == covid:
        print('ban da bi covid 19')
        # pass
    elif max1 == pneumonia:
        print('ban da bi ung thu phoi')
    else: 
        print('phoi binh thuong nhe, yen tam')
        # pass