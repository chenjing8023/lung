import tensorflow as tf
import numpy as np
import glob

class Hyperparams:
    # parameters for image
    image_height = 40
    image_width = 40
    image_depth = 1
    image_pixel_dtype = np.uint8
    image_pixel_bytes = 1
    image_bytes = image_height * image_width * image_depth * image_pixel_bytes

    # parameters for label
    label_dtype = np.uint8 
    label_bytes = 2

    # other parameters
    num_classes = 2
    num_threads = 1
    batch_size = 64

HP = Hyperparams # alias

def read_record(filename_queue):
    class Record(object):
        pass

    result = Record()

    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = HP.label_bytes + HP.image_bytes

    # Read a record, getting filenames from the filename_queue
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, np.uint8)
    label_bytes = tf.slice(record_bytes, begin=[0], size=[HP.label_bytes])
    image_bytes = tf.slice(record_bytes, begin=[HP.label_bytes], size=[HP.image_bytes])

    # Convert label and image to its original type
    label_bytes = tf.reshape(label_bytes, [-1, HP.label_bytes])
    label = tf.bitcast(label_bytes, HP.label_dtype)
    image_bytes = tf.reshape(image_bytes, [-1, HP.image_pixel_bytes])
    image = tf.bitcast(image_bytes, HP.image_pixel_dtype)

    # Do proper reshape

    result.label = tf.reshape(label, [HP.label_bytes])
    result.image = tf.reshape(image, [HP.image_height * HP.image_width * HP.image_depth])
    result.image = tf.cast(result.image, tf.float32)
    return result


def generate_batch(tensors, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = HP.num_threads
    if shuffle:
        batched_tensors = tf.train.shuffle_batch(
            tensors,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        batched_tensors = tf.train.batch(
            tensors,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    return batched_tensors

def inputs(datadir, is_training=True):
    filenames = glob.glob(datadir)
    filenameQueue = tf.train.string_input_producer(filenames)
    read_input = read_record(filenameQueue)
    image = read_input.image
    label = read_input.label

    # Generate a batch of images and labels by building up a queue of examples.
    min_queue_examples = HP.batch_size * 32
    return generate_batch([image, label], min_queue_examples, HP.batch_size, shuffle=is_training)

datadir = "D:\\chenjing\\lung\\dest\\*.bin"
'''
with tf.Graph().as_default():
    data = inputs(datadir,False)
    sv = tf.train.Supervisor(logdir='save')
    with sv.managed_session() as sess:
        images, labels = sess.run(data)
        print(images)
        print(labels)
        # data2 = sess.run(data[1])
 '''       # print(data2)
