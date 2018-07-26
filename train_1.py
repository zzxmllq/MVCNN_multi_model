import tensorflow as tf
import reader
import tensorflow.contrib.slim as slim
import alexnet
import loss

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 16, 'Number of blocks in each batch')
flags.DEFINE_string('root_path', '/media/zzxmllq/0002605F0001462E/mvcnn/',
                    'Training data file path.')
flags.DEFINE_integer('data_size', 3183, 'Number of images')
flags.DEFINE_integer('n_classes', 40, 'Number of classes')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('keep_prob', 0.5, 'the rate of keep_prob')
flags.DEFINE_integer('max_epoch', 150, 'Number of epoch')
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
tfrecord_filename = FLAGS.root_path + 'train.tfrecords'
MOVING_AVERAGE_DECAY = 0.9999


def train():
    filename_queue = tf.train.string_input_producer([tfrecord_filename])
    images, labels = reader.read_and_decode(filename_queue=filename_queue, batch_size=FLAGS.batch_size)
    with tf.device('/gpu:0'):
        global_step = slim.create_global_step()
    with tf.device('/cpu:0'):
        num_batches_per_epoch = FLAGS.data_size / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        lrn_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
    with tf.device('/gpu:0'):
        fc8 = alexnet.model(input_data=images, n_classes=FLAGS.n_classes, keep_prob=FLAGS.keep_prob)
        total_loss = loss.get_loss(input_data=fc8, grdtruth=labels)
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        with tf.control_dependencies([loss_averages_op]):
            optimizer = tf.train.AdamOptimizer(lrn_rate)
        train_step = optimizer.minimize(loss=total_loss, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_step = tf.no_op(name='train')
        prediction = alexnet.classify(fc8)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for j in range(FLAGS.max_epoch):
            for i in range(200):
                _, loss_value, pre, grd = sess.run([train_step, total_loss, prediction, labels])
                print str(j + 1) + ' epoch' + ' ' + str(i) + ' minibatch' + ':' + str(loss_value)
                print str(j + 1) + ' epoch' + ' ' + str(i) + ' minibatch' + ':' + str(pre) + " " + str(grd)
        coord.request_stop()
        coord.join(threads)


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
