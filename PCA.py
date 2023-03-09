from readData import read_data, data_Split, eigen
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# this function get data and alpha then perform PCA and return transformed matrix(n, r)
def PCA(data, alpha, eigvalues, eigvectors):
    # compute centralized data matrix , covariance matrix , eigenvalues and eigenvectors
    mean_vector = np.mean(data, axis=0)
    z = data - mean_vector
    eigenvalues, eigenvectors = eigvalues, eigvectors
    # sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # get the value of r according alpha
    eigenvalues_sum = 0
    r_sum = 0
    r = 0
    for i in range(len(eigenvalues)):
        eigenvalues_sum += eigenvalues[i]
    while eigenvalues_sum * alpha > r_sum:
        r_sum += eigenvalues[r]
        r += 1
    # compute transformed data matrix
    p = eigenvectors[:, :r]
    transformed_data = np.dot(z, p)
    return mean_vector, p, transformed_data


# this function reduce the dimensions of testing data
def organizeTest(mean_vector, p, d_test):
    z = d_test - mean_vector
    transformed_test = np.dot(z, p)
    return transformed_test


# get predict labels vector for testing data
def get_predict(d_samples, d_test, y_samples , k):
    classifier = KNeighborsClassifier(n_neighbors=k , weights='distance')
    classifier.fit(d_samples, y_samples)
    y_pred = classifier.predict(d_test)
    return y_pred


def get_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def get_accuracy_report():
    data = read_data()
    d_samples, d_test, y_samples, y_test = data_Split(data[0], data[1])
    alpha = np.array([0.8, 0.85, 0.9, 0.95])
    k = np.array([1,3,5,7])
    eigenvalues, eigenvectors = eigen(d_samples)
    output = []
    for i in range(4):
        row = []
        for j in range(4):
            mean_vector, p, trans_data = PCA(d_samples, alpha[i], eigenvalues, eigenvectors)
            d_test_trans = organizeTest(mean_vector, p, d_test)
            y_pre = get_predict(trans_data, d_test_trans, y_samples , k[j])
            accuracy = get_accuracy(y_test, y_pre)
            row.append(accuracy)
        output.append(row)
    return np.array(output)



out = get_accuracy_report()
print(out)