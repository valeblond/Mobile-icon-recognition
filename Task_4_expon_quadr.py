import tensorflow as tf
from tensorflow import keras

from definitions import *
from preprocessing import load_datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from kernel_types import Exponential,Quadratic

# Load datasets
train_ds, test_ds, val_ds = load_datasets()

# Create model
ALPHA = 1.0  # controls the width of the network
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
TARGET_SIZE = (192, 192)
BATCH_SIZE = 256

# Create simpler model
simpler_base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE,
                                                       include_top = False,
                                                       alpha       = ALPHA,
                                                       weights     = 'imagenet')

# Remove block_16 layers from base model
last_layer_old = simpler_base_model.get_layer('block_15_project_BN').output
x = keras.layers.Conv2D(**simpler_base_model.get_layer('Conv_1').get_config())(last_layer_old)
x = keras.layers.BatchNormalization(**simpler_base_model.get_layer('Conv_1_bn').get_config())(x)
last_layer_new = keras.layers.ReLU(**simpler_base_model.get_layer('out_relu').get_config())(x)

simpler_base_model = tf.keras.Model(inputs = simpler_base_model.layers[0].input,
                                    outputs = last_layer_new,
                                    name='Simpler')

def image_preprocessing(subdir, validation_split=0.0):
    data_dir = subdir
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_split)
    train_datagen = train_datagen.flow_from_directory(data_dir,
                                          target_size=TARGET_SIZE,
                                          color_mode='rgb',
                                          classes=None,
                                          class_mode='categorical',
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          interpolation='nearest')
    return train_datagen

train_generator = image_preprocessing("split_dataset/train_data/", 0.2)
test_generator = image_preprocessing("split_dataset/test_data/")
history = simpler_base_model.predict_generator(train_generator)

kernels = [Exponential(),Quadratic()]

for kernel in kernels:
    svm = SVC(kernel=kernel)
    svm.fit(history, train_generator.classes)

    x_test = simpler_base_model.predict(test_generator)
    y_pred = svm.predict(x_test)

    cm = confusion_matrix(test_generator.classes, y_pred)
    print(cm)

