import tensorflow as tf
import numpy as np
import glob

dtype = np.int8
size = 1
IMAGE_SIZE = 40
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
NUM_THREADS = 1

batch_size = 2


def read(filename_queue):
    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    label_bytes = 1  # 2 for CIFAR-100
    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = 1
    image_bytes = result.height * result.width * result.depth

    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = (label_bytes + image_bytes) * size

    # Read a record, getting filenames from the filename_queue
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, dtype)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    result.label = tf.reshape(result.label,[1])
    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.height * result.width])
    result.uint8image = depth_major
    print(result.uint8image.shape)
    print(result.label.shape)
    # Convert from [depth, height, width] to [height, width, depth].
    #result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    #tf.reshape(result.uint8image, shape=[-1])
    #tf.contrib.layers.flatten(result.uint8image)
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = NUM_THREADS
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    # Display the training images in the visualizer.
    #tf.summary.image('images', images)
    return images, label_batch


def distorted_inputs(datadir):
    directory = glob.glob(datadir)
    filenameQueue = tf.train.string_input_producer(directory)
    read_input = read(filenameQueue)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    #reshaped_image = tf.reshape(reshaped_image,[1600])
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    # Subtract off the mean and divide by the variance of the pixels.
    # float_image = tf.image.per_image_standardization(reshaped_image)
    float_image = reshaped_image
    # Set the shapes of tensors.
    #float_image.set_shape([1])
    #read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)



datadir = "/Users/chenjing/PycharmProjects/mytask/dest/*.bin"
with tf.Graph().as_default():
    data = distorted_inputs(datadir)
    sv = tf.train.Supervisor(logdir='save')
    with sv.managed_session() as sess:
        data = sess.run(data)
        print(data[0].shape)
        print(data[1].shape)
        # data2 = sess.run(data[1])
        # print(data2)