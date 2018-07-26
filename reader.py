import tensorflow as tf


def read_and_decode(filename_queue, batch_size,shuffle_batch=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.float32)
    # image = tf.reshape(image, [12, 227, 227, 3])
    image=tf.reshape(image,[12,224,224,3])
    label=tf.cast(features['label'],tf.int32)
    if shuffle_batch:
        images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=6400,num_threads=64,
                                                min_after_dequeue=256)
    else:
        images, labels = tf.train.batch([image, label], batch_size=batch_size, capacity=8000, num_threads=16)
    return images,labels
