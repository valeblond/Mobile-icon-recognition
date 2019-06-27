#create folders: 'split_dataset', 'split_dataset/train_data', 'split_dataset/test_data' in folder 'snr' before
#running this preprocessing.

#Creating training and testing folders that will contain all classes and some pictures of each class
import os
import random
import shutil

old_path = 'raw_data'
new_train_path = 'split_dataset/train_data'
new_test_path = 'split_dataset/test_data'

subdirs = [f.name for f in os.scandir(old_path) if f.is_dir()]
split = 0.7 #splitting each folder(train=0,7*files, test=0.3*files)

for subdir in subdirs:
    subdir_path = os.path.join(old_path, subdir)

    files = []
    files += [os.path.join(subdir_path, f.name) for f in os.scandir(subdir_path) if f.is_file()]
    not_jpg = [f for f in files if not f.endswith(".jpg")]
    files = [file for file in files if file not in not_jpg]
    random.shuffle(files) #to get files of each folder in different order
    num_files = len(files)

    train_files = files[0:int(split*num_files)]
    test_files = files[int(split*num_files):]

    train_dir = 'split_dataset/train_data/' + subdir
    test_dir = 'split_dataset/test_data/' + subdir

    train_path = os.path.join(new_train_path, subdir)
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    test_path = os.path.join(new_test_path, subdir)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for file in train_files:
        shutil.copy(file, train_dir)
    for file in test_files:
        shutil.copy(file, test_dir)
