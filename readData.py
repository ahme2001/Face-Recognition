import numpy as np
# take a path of pgm image and return vector represent the image data
def read_image(image_path):
    f = open(image_path, 'rb')
    # Skip the first 3 lines
    f.readline()
    f.readline()
    f.readline()
    raster = []
    # read the data
    for h in range(112 * 92):
        row = [ord(f.read(1))]
        raster.append(row)
    return np.array(raster).T


# this function read all image files and return matrix of images data and label vector
def read_data():
    d = np.zeros((400, 112 * 92))
    y = np.zeros(400)
    for i in range(40):
        for j in range(10):
            path = r"C:\Users\lenovo\Desktop\Term8\pattern\labs\lab1\archive\s" + str(i + 1) + "\\" + str(
                j + 1) + ".pgm"

            d[i * 10 + j, :] = np.array(read_image(path))
            # Add label to label vector
            y[i * 10 + j] = i + 1
    return np.array(d), np.array(y)


# function get data matrix and label vector and split each one to two matrix , one as a samples and one for testing
# splitting data 50:50
def data_Split(data, y):
    y_samples = []
    y_test = []
    d_samples = []
    d_test = []
    i = 0
    while i < len(y) - 1:
        y_samples.append(y[i])
        y_test.append(y[i + 1])
        d_samples.append(data[i])
        d_test.append(data[i + 1])
        i += 2

    return np.array(d_samples), np.array(d_test), np.array(y_samples), np.array(y_test)

# function get data matrix and label vector and split each one to two matrix , one as a samples and one for testing
# to split data into 70:30
def data_Split2(data, y):
    y_samples = []
    y_test = []
    d_samples = []
    d_test = []
    i = 0
    while i < len(y) :
        if i%10 <7 :
            y_samples.append(y[i])
            d_samples.append(data[i])
        else:
            y_test.append(y[i])
            d_test.append(data[i])
        i +=1
    return np.array(d_samples), np.array(d_test), np.array(y_samples), np.array(y_test)


data , labels = read_data()
d_samples, d_test, y_samples, y_test = data_Split(data,labels)