import numpy as np
import scipy
from imports import *
from readData import *
from PCA import *

def read_non_face_data():
    d = []
    for i in range(220):
        # img = cv2.imread('/content/drive/MyDrive/archive/r (' + str(i+1) + ')'+'.jpg', 0)
        img = cv2.imread('/content/drive/MyDrive/new non face/r (' + str(i+1) + ')'+'.jpg', 0)
        img_vector = np.array(img, dtype='float64').flatten()
        d.append(img_vector)
    return np.array(d)


def limit_data(non_faces,faces,limit):
    new_non_faces = np.array(non_faces[:limit])
    new_data = np.concatenate((new_non_faces, faces), axis=0)
    new_data_labels = np.concatenate((np.zeros(new_non_faces.shape[0],), np.ones(faces.shape[0])), axis=0)
    return data_Split(new_data, new_data_labels)


def Non_faces_accuracy(y_test, y_pre):
    failureIdx = []
    successIdx = []
    for i in range(len(y_pre)):
        if y_test[i] == y_pre[i]:
            successIdx.append(i)
        else:
            failureIdx.append(i)
    return np.array(failureIdx), np.array(successIdx), (len(successIdx) / len(y_pre))

def LDA(data,label):
    # Calculate the mean vector for every class Mu1, Mu2, ..., Mu40.
    class_means = []
    for c in np.unique(label):
        class_means.append(np.mean(data[label == c], axis=0))
    class_means = np.array(class_means)

    # Calculate the overall mean vector μ as the mean of all samples
    mean_vector = np.mean(data, axis=0)

    # Calculate the between-class scatter matrix Sb
    Sb = np.zeros((10304, 10304))
    mean_vector = mean_vector.reshape(-1, 1)
    for mean_class , n in zip(class_means,label):
        mean_class = mean_class.reshape(-1,1)
        diff_vector = np.subtract(mean_class,mean_vector)
        mul = np.dot(diff_vector,diff_vector.T)
        Sb += (data.shape[0] * mul)

    # center class matrices and sum scatter matrix
    S = np.zeros((10304,10304))
    for c, mean_vec in zip(np.unique(label), class_means):
        class_indices = np.where(label == c)[0]
        z_matrix = data[class_indices] - mean_vec
        class_scatter = np.dot(z_matrix.T,z_matrix)
        S += class_scatter

    s_inv = scipy.linalg.pinv(S)
    total_matrix = np.dot(s_inv,Sb)
    eigenValue, eigenVector = scipy.linalg.eig(total_matrix)
    eigenValue = np.real(eigenValue)
    eigenVector = np.real(eigenVector)
    idx = np.argsort(eigenValue)[::-1]
    eigenvectors = eigenVector[:,idx]
    U = eigenvectors[:,:1]
    return np.real(U)


def visualize_non_faces(index,array):
    fig, axes = plt.subplots(1, min(len(index),20), figsize=(30, 30),
                            subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(array[index[i]].reshape(112, 92), cmap='gray')
    plt.show()

def visualize_faces(array):
    fig, axes = plt.subplots(1, min(len(array),20), figsize=(30, 30),
                            subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(array[i].reshape(112, 92), cmap='gray')
    plt.show()

def Non_faces_Pca_report(non_faces, faces):
    limits = [200,400,600,800]
    output = []
    for i in range(4):
        d_samples, d_test, y_samples, y_test = limit_data(non_faces, faces, limits[i])
        eigenvalues, eigenvectors = eigen(d_samples)
        mean_vector, p, trans_data = PCA(d_samples, 0.95, eigenvalues, eigenvectors)
        d_test_trans = organizeTest(mean_vector, p, d_test)
        y_pre = get_predict(trans_data, d_test_trans, y_samples, 1)
        failure ,success ,accuracy  = Non_faces_accuracy(y_test, y_pre)
        print("At non faces = "+str(limits[i]))
        print("success")
        visualize_non_faces(success,d_test)
        print("failure")
        visualize_non_faces(failure,d_test)
        print("with accuracy = " + str(accuracy))
        output.append(accuracy)
    plt.scatter(limits,output)
    plt.show() 
    

def Non_faces_LDA_report(non_faces, faces):
    limits = [200,400,600,800]
    output = []        
    for i in range(4):
        d_samples, d_test, y_samples, y_test = limit_data(non_faces, faces, limits[i])
        U = LDA(d_samples, y_samples)
        sample = np.dot(d_samples ,U)
        test = np.dot(d_test ,U)
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(sample, y_samples)
        y_pre = classifier.predict(test)
        failure ,success ,accuracy  = Non_faces_accuracy(y_test, y_pre)
        print("At non faces = "+str(limits[i]))
        print("success")
        visualize_non_faces(success,d_test)
        print("failure")
        visualize_non_faces(failure,d_test)
        print("with accuracy = " + str(accuracy))
        output.append(accuracy)
    plt.scatter(limits,output)
    plt.show()