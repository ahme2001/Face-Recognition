# Use a simple classifier (first Nearest Neighbor to determine the class labels).
from imports import *
from readData import *
from LDA import *

# Project the training set, and test sets separately using the same projection matrix U.
def projection(data,eigenvec):
    return np.dot(data,eigenvec)

# Use a simple classifier (first Nearest Neighbor to determine the class labels).
def get_accuracy(sample , test , y_sample , y_test):
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(sample, y_sample)
    y_pred = classifier.predict(test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def classify( N, sample, test, y_test, y_samples):
    classifier = KNeighborsClassifier(n_neighbors=N)
    classifier.fit(sample, y_samples)
    y_pred = classifier.predict(test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

w = LDA(d_samples , y_samples)
trans_data = projection(d_samples , w)
trans_test = projection(d_test , w)
print(get_accuracy(trans_data , trans_test , y_samples , y_test))

xtemp=[]
ytemp=[]
for i in range(1,8,2):
    accuracy = classify(i,trans_data , trans_test,y_test,y_samples)
    xtemp.append(i)
    ytemp.append(accuracy)

plt.scatter(xtemp,ytemp)
plt.show()