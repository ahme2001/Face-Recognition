import numpy as np
from PIL import Image
import os

def read_images():
    data = []
    labels = []
    for i in range(1, 41):
        for j in range(1, 11):
            # path where data stored , you must change it according to your computer
            img_path = r"/content/drive/MyDrive/archive/s" + str(i) + "/" + str(j) + ".pgm"
            img = Image.open(img_path)
            img_vector = np.array(img).flatten()
            data.append(img_vector)
            labels.append(i)
    return np.array(data), np.array(labels)

# function get data matrix and label vector and split each one to two matrix , one as a samples and one for testing
def data_Split(data, y):
    d_test = data[1::2 , :]     # odd data
    d_samples = data[::2 , :]   # even data
    l_samples = y[::2]          # label for even data
    l_test = y[1::2]            # label for odd data

    return d_samples, d_test, l_samples, l_test 

data ,y= read_images()
d_samples, d_test, y_samples, y_test = data_Split(data,y)


# print(d_samples)
# print(d_test)
# print(y_samples)
# print(y_test)