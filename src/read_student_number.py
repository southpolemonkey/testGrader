import cv2
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree

# test student number image
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
# cv2.imshow('student id', image)

GROUND_TRUTH = [5, 6, 7, 8, 9, 8, 7, 7, 2]

# Now load the data to knn model
knn = cv2.ml.KNearest_create()
with np.load('knn_data.npz') as data:
    # print(data.files)
    train = data['train']
    train_labels = data['train_labels']
    train_labels = np.ravel(train_labels)  # change the label to 1d array
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
# random forest
clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
clf.fit(train, train_labels)
# svm
clf2 = svm.SVC(gamma=0.01, kernel="rbf")
clf2.fit(train, train_labels)
# knn
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train, train_labels)
# decision tree
clf3 = tree.DecisionTreeClassifier()
clf3.fit(train, train_labels)

# Resize the test data
image = cv2.resize(image, (180, 20))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imwrite('bwid.png', thresh)

# vertically split student number into 9 digits
cells = [np.hsplit(thresh, 9)]
x = np.array(cells)


def model_accuracy(test_file):
    matches = test_file == GROUND_TRUTH
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/test_file.size
    return accuracy


# feed test data into trained model
test = x.reshape(-1, 400).astype(np.float32)
ret, results, neighbours, dist = knn.findNearest(test, k=5)
# print('Result:')
print("KNN ====>")
print(np.reshape(results, -1))
print(model_accuracy(np.reshape(results, -1)))
print("\n")
print("Random Forest ====>")
print(clf.predict(test))
print(model_accuracy(clf.predict(test)))
print("\n")
print("SVM ====>")
print(clf2.predict(test))
print(model_accuracy(clf2.predict(test)))
print("\n")
print("Scikit KNN ====>")
print(neigh.predict(test))
print(model_accuracy(neigh.predict(test)))
print("\n")
print("Decision tree ====>")
print(clf3.predict(test))
print(model_accuracy(clf3.predict(test)))