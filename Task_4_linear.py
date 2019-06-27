import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from definitions import *
from PIL import Image
from skimage.color import rgb2grey

# Create list of files and labels
subdirs = [f.name for f in os.scandir(raw_data_dir) if f.is_dir()]

files = []
for subdir in subdirs:
    subdir_path = os.path.join(raw_data_dir, subdir)
    files += [os.path.join(subdir_path, f.name) for f in os.scandir(subdir_path) if f.is_file()]

# Remove not images from dataset
not_jpg = [f for f in files if not f.endswith(".jpg")]
files = [file for file in files if file not in not_jpg]


# Sort files to maintain order
files = sorted(files)
#print(files)

labels = [file.split("\\")[-2] for file in files]

#Encode labels
label2index = dict((label, index) for index, label in enumerate(sorted(set(labels))))
encoded_labels = [label2index[label] for label in labels]
# print(files[10560:10570])
# print()
# print(labels[10560:10570])
# print()
# print(encoded_labels[10560:10570])

#Split files into training, testing and validation
NUMBER_OF_FILES  = len(files)
NUMBER_OF_LABELS = len(label2index)

def get_image(row_id):
    img = Image.open(row_id)
    return np.array(img)

def transform(img):
    img = img.flatten()
    return img

def get_features(files):
    #2D tensor variable
    features = []

    for file in files:
        #Transformation to array
        array = get_image(file)
        grey_array = rgb2grey(array)
        #Transformation to 1D array
        grey_array = transform(grey_array)
        features.append(grey_array)
        print(features)

    features_matrix = np.array(features)
    return features_matrix

features_matrix = get_features(files)
# print(features_matrix)
print(features_matrix.shape)

# bombus = get_image("D:/study/python/myNeuralNetworks/6sem/raw_data/_negative/_00a5b39e6f9b6826.jpg")
# grey_bombus = rgb2grey(bombus)
# grey_bombus = transform(grey_bombus)
# print(bombus)
# print()
# print(grey_bombus)
# print(type(grey_bombus))

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
bees_stand = ss.fit_transform(features_matrix)

pca = PCA(n_components=500)
# use fit_transform to run PCA on our standardized matrix
bees_pca = ss.fit_transform(bees_stand)
# look at new shape
print('PCA matrix shape is: ', bees_pca.shape)

X = pd.DataFrame(bees_pca)
y = pd.Series(encoded_labels)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.3,
                                                    random_state=1234123)

# define support vector classifier
svm = SVC(kernel='linear', probability=True, random_state=42)

# fit model
svm.fit(X_train, y_train)

# generate predictions
y_pred = svm.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)


