import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import readData as data

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
        mul = ((diff_vector).dot((diff_vector).T))
        Sb += (5 * mul)

    # center class matrices and sum scatter matrix
    S = np.zeros((10304,10304))
    for c, mean_vec in zip(np.unique(label), class_means):
        class_indices = np.where(label == c)[0]
        z_matrix = data[class_indices] - mean_vec
        class_scatter = z_matrix.T.dot(z_matrix)
        S += class_scatter

    s_inv = np.linalg.pinv(S)
    total_matrix = s_inv.dot(Sb)
    eigenValue, eigenVector = np.linalg.eig(total_matrix)
    eigenValue = np.real(eigenValue)
    eigenVector = np.real(eigenVector)
    eig_pairs = [(np.abs(eigenValue[i]), eigenVector[:, i]) for i in range(len(eigenValue))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    # Construct projection matrix U with 39 dominant eigenvectors
    U = np.hstack([eig_pairs[i][1].reshape(-1, 1) for i in range(39)])
    return np.real(U)

W = LDA(data.d_samples,data.y_samples)

# Project the training set, and test sets separately using the same projection matrix U.
def projection(data,eigenvec):
    return data.dot(eigenvec)

sample = projection(data.d_samples,W)
test = projection(data.d_test,W)

# Use a simple classifier (first Nearest Neighbor to determine the class labels).
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(sample, data.y_samples)
y_pred = classifier.predict(test)

def get_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

accuracy = get_accuracy(data.y_test,y_pred)
print(accuracy)