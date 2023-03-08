import numpy as np

def LDA(data,label):
    # Calculate the mean vector for every class Mu1, Mu2, ..., Mu40.
    class_means = []
    for c in np.unique(label):
        class_means.append(np.mean(data[label == c], axis=0))
    class_means = np.array(class_means)

    # Calculate the overall mean vector μ as the mean of all samples
    mean_vector = np.mean(data, axis=0)
    # print(sorted(mean_vector))

    # Calculate the between-class scatter matrix Sb
    Sb = np.zeros((10304, 10304))
    mean_vector = mean_vector.reshape(-1, 1)
    for mean_class , n in zip(class_means,label):
        mean_class = mean_class.reshape(-1,1)
        diff_vector = np.subtract(mean_class,mean_vector)
        mul = 5 * ((diff_vector).dot((diff_vector).T))
        Sb += mul

    # center class matrices and sum scatter matrix
    S = np.zeros((10304,10304))
    for c, mean_vec in zip(np.unique(label), class_means):
        class_indices = np.where(label == c)[0]
        z_matrix = data[class_indices] - mean_vec
        class_scatter = z_matrix.T.dot(z_matrix)
        S += class_scatter

    # get eigenvalue and eigenvector
    s_inv = np.linalg.inv(S)
    total_matrix = s_inv.dot(Sb)
    eigenValue , eigenVector = np.linalg.eig(total_matrix)
    eigenVector = np.array(eigenVector)
    eig_pairs = [(np.abs(eigenValue[i]), eigenVector[:, i]) for i in range(len(eigenValue))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # Construct projection matrix U with 39 dominant eigenvectors
    U = np.hstack([eig_pairs[i][1].reshape(-1, 1) for i in range(39)])
    return U
