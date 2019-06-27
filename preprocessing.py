import tensorflow as tf
from definitions import *



def load_datasets(batch_size=BATCH_SIZE, prefetch_size=PREFETCH_SIZE, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE):
    train_images_ds, test_images_ds, val_images_ds = load_images_datasets()
    train_labels_ds, test_labels_ds, val_labels_ds = load_labels_datasets()
    
    train_ds = zip_dataset(train_images_ds, train_labels_ds)
    test_ds  = zip_dataset( test_images_ds,  test_labels_ds)
    val_ds   = zip_dataset(  val_images_ds,   val_labels_ds)

    train_ds = apply_train_tweaks(train_ds, batch_size, prefetch_size, shuffle_buffer_size)
    test_ds  = apply_test_tweaks(  test_ds, batch_size, prefetch_size)
    val_ds   = apply_val_tweaks(    val_ds, batch_size, prefetch_size)

    return train_ds, test_ds, val_ds

    
def load_images_datasets():
    train_images_ds = tf.data.TFRecordDataset(train_images_file)
    test_images_ds  = tf.data.TFRecordDataset( test_images_file)
    val_images_ds   = tf.data.TFRecordDataset(  val_images_file)
    
    train_images_ds = parse_images_dataset(train_images_ds)
    test_images_ds  = parse_images_dataset( test_images_ds)
    val_images_ds   = parse_images_dataset(  val_images_ds)
    
    return train_images_ds, test_images_ds, val_images_ds


def load_labels_datasets():
    train_labels_ds = tf.data.TFRecordDataset(train_labels_file)
    test_labels_ds  = tf.data.TFRecordDataset( test_labels_file)
    val_labels_ds   = tf.data.TFRecordDataset(  val_labels_file)
    
    train_labels_ds = parse_labels_dataset(train_labels_ds)
    test_labels_ds  = parse_labels_dataset( test_labels_ds)
    val_labels_ds   = parse_labels_dataset(  val_labels_ds)
    
    return train_labels_ds, test_labels_ds, val_labels_ds


def parse_images_dataset(dataset):
    dataset = dataset.map(parse_images, num_parallel_calls=4)
    return dataset
    
    
def parse_labels_dataset(dataset):
    dataset = dataset.map(parse_labels, num_parallel_calls=4)
    return dataset


def apply_train_tweaks(dataset, batch_size, prefetch_size, shuffle_buffer_size):
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_size)
    
    return dataset


def apply_test_tweaks(dataset, batch_size, prefetch_size):
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_size)
    
    return dataset


def apply_val_tweaks(dataset, batch_size, prefetch_size):
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_size)
    dataset = dataset.repeat()
    
    return dataset


def zip_dataset(dataset_1, dataset_2):
    zipped = tf.data.Dataset.zip((dataset_1, dataset_2))
    return zipped    
    
    
def parse_labels(label):
    label = tf.io.parse_tensor(label, out_type=tf.int32)
    return label


def parse_images(image):
    image = tf.io.parse_tensor(image, out_type=tf.string)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1  # Normalize to [-1, 1]

    return image

def time_footprint():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
