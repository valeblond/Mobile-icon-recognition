dataset_name = 'testdotai/common-mobile-web-app-icons'
raw_data_dir = 'raw_data'
output_datasets_dir = 'datasets'

train_images_file = output_datasets_dir + '/' + 'train_images.tfrec'
test_images_file  = output_datasets_dir + '/' +  'test_images.tfrec'
val_images_file   = output_datasets_dir + '/' +   'val_images.tfrec'

train_labels_file = output_datasets_dir + '/' + 'train_labels.tfrec'
test_labels_file  = output_datasets_dir + '/' +  'test_labels.tfrec'
val_labels_file   = output_datasets_dir + '/' +   'val_labels.tfrec'

LOG_DIR = 'logs'
MODEL_DIR = 'models'

IMG_SIZE = 192
NUMBER_OF_FILES = 153378
NUMBER_OF_LABELS = 106

BATCH_SIZE=256
PREFETCH_SIZE=1
SHUFFLE_BUFFER_SIZE=1000

TEST_SPLIT_FACTOR = 0.2
VAL_SPLIT_FACTOR  = 0.2

TEST_FILES  = int(NUMBER_OF_FILES * TEST_SPLIT_FACTOR)
VAL_FILES   = int((NUMBER_OF_FILES - TEST_FILES) * VAL_SPLIT_FACTOR)
TRAIN_FILES = NUMBER_OF_FILES - TEST_FILES - VAL_FILES

