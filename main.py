import os
import random
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.model_selection import train_test_split


# ----------load and change the data to grayscale---------
def load_data():
    images = []
    labels = []
    folder_path = "./image"
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        for filename in os.listdir(category_path):
            if not filename.endswith('.jpg'):
                continue
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, 0)
            images.append(img)
            labels.append(category)
    return np.array(images), np.array(labels)


data, labels = load_data()

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, shuffle=True)

# random.shuffle(labels)

"""
plt.imshow(data[0])
plt.show()
"""


# -------------GLCM------------------
def GLCM(img, angle, distance):
    maximum = np.max(img)
    print(maximum)
    co_occurence_matrix = [[0] * maximum] * maximum
# /matrix
    for i in range(len(img)):
        for j in range(len(img[0])):
            if 0 <= angle < 45:
                if j + distance < len(img[0]) and img[i][j + distance] < len(co_occurence_matrix) and img[i][j] < len(co_occurence_matrix):
                    # print(img[i][j + distance])
                    co_occurence_matrix[img[i][j]][img[i][j + distance]] += 1

            elif 45 <= angle < 90:
                new_i = i - distance
                new_j = j + distance
                if new_i >= 0 and new_j < len(img[0]):
                    co_occurence_matrix[img[i][j]][img[new_i][new_j]] += 1

            elif 90 <= angle < 135:
                if i + distance >= 0:
                    co_occurence_matrix[img[i][j]][img[i - distance][j]] += 1

            elif 135 <= angle < 180:
                if i - distance >= 0 and j - distance >= 0:
                    co_occurence_matrix[img[i][j]][img[i - distance][j - distance]] += 1

            elif 180 <= angle < 225:
                if j - distance >= 0:
                    co_occurence_matrix[img[i][j]][img[i][j - distance]] += 1

            elif 225 <= angle < 270:
                new_i = i + distance
                new_j = j - distance
                if new_j >= 0 and new_i < len(img):
                    co_occurence_matrix[img[i][j]][img[new_i][new_j]] += 1

            elif 270 <= angle < 315:
                if i + distance < len(img):
                    co_occurence_matrix[img[i][j]][img[i + distance][j]] += 1

            elif 315 <= angle < 360:
                if i + distance < len(img) and j + distance < len(img[0]):
                    co_occurence_matrix[img[i][j]][img[i + distance][j + distance]] += 1

    summation = 0
    for i in range(len(co_occurence_matrix)):
        for j in range(len(co_occurence_matrix[0])):
            summation += co_occurence_matrix[i][j]

    for i in range(len(co_occurence_matrix)):
        for j in range(len(co_occurence_matrix[0])):
            if summation > 0:
                co_occurence_matrix[i][j] = int(co_occurence_matrix[i][j] / summation)

# -------------calculate the accuracy---------------


    # calculate the features-------------------------------
    mean_i = 0
    mean_j = 0
    variance_i = 0
    variance_j = 0
    # 1
    correlation = 0
    # 2
    contrast = 0
    # 3
    energy = 0
    # 4
    homogeneity = 0

    for i in range(len(co_occurence_matrix)):
        for j in range(len(co_occurence_matrix[0])):
            contrast = contrast + (pow(abs(i - j), 2) * co_occurence_matrix[i][j])
            # finding the mean
            mean_i += (i * co_occurence_matrix[i][j])
            mean_j += (j * co_occurence_matrix[i][j])
            # finding the variance
            variance_i += co_occurence_matrix[i][j] * pow(i - mean_i, 2)
            variance_j += co_occurence_matrix[i][j] * pow(j - mean_j, 2)
            # calculate the correlation
            if co_occurence_matrix[i][j] == 0:
                correlation = 0
                homogeneity = 0
            else:
                correlation = correlation + (((i - mean_i) * (j - mean_j) * co_occurence_matrix[i][j]) / (variance_i * variance_j))
                # calculate the homogeneity
                homogeneity = homogeneity + (co_occurence_matrix[i][j] * math.log(2, co_occurence_matrix[i][j]))
            # calculate the energy
            energy = energy + pow(co_occurence_matrix[i][j], 2)

    feature_vector = [contrast, correlation, energy, homogeneity]

    return feature_vector


# ----------calculate the GLCM for all images-------------
ang = int(input("Enter the theta : "))
dis = int(input("Enter a distance : "))
list_image_train = []
for k in range(len(train_data)):
    vec = GLCM(train_data[k], ang, dis)
    vec2 = [vec[0], vec[1], vec[2], vec[3], train_labels[k]]
    list_image_train.append(vec2)

minimum = 0
accuracy = 0
temp2 = ""
predict = []
for k in range(len(test_data)):
    for m in range(len(list_image_train)):
        vec = GLCM(train_data[k], ang, dis)
        temp = pow((vec[0] - list_image_train[m][0]), 2) + pow((vec[1] - list_image_train[m][1]), 2) + pow((vec[2] - list_image_train[m][2]), 2) + pow((vec[3] - list_image_train[m][3]), 2)
        temp = pow(temp, 0.5)
        if m == 0:
            minimum = temp
            temp2 = list_image_train[m][4]
        elif temp < minimum:
            minimum = temp
            temp2 = list_image_train[m][4]
    predict.append(test_labels[k])
    if test_labels[k] == temp2:
        accuracy += 1

accuracy = sklearn.metrics.accuracy_score(test_labels, predict)
#accuracy = (accuracy/len(test_data)) * 100

print(accuracy, "%")
