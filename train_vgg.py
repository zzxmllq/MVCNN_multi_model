import tensorflow as tf
import reader
import tensorflow.contrib.slim as slim
import vgg_m
import loss

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 1, 'Number of blocks in each batch')
flags.DEFINE_string('root_path', '/media/zzxmllq/0002605F0001462E/mvcnn/',
                    'Training data file path.')
flags.DEFINE_integer('data_size', 3183, 'Number of images')
flags.DEFINE_integer('n_classes', 40, 'Number of classes')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('keep_prob', 0.5, 'the rate of keep_prob')
flags.DEFINE_integer('max_epoch', 150, 'Number of epoch')
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
tfrecord_filename = FLAGS.root_path + 'train_224.tfrecords'
model_path='/media/zzxmllq/0002605F0001462E/mvcnn/model_vgg/vggnet.ckpt'

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
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.device('/gpu:0'):
        optimizer = tf.train.AdamOptimizer(lrn_rate)
        fc8 = vgg_m.model(input_data=images, n_classes=FLAGS.n_classes, keep_prob=FLAGS.keep_prob)
        losses = loss.get_loss(input_data=fc8, grdtruth=labels)
        train_step = optimizer.minimize(loss=losses, global_step=global_step)
        prediction = vgg_m.classify(fc8)
    with tf.device('/cpu:0'):
        saver=tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for j in range(FLAGS.max_epoch):
            for i in range(200):
                _, loss_value, pre,grd = sess.run([train_step, losses, prediction,labels])
                print str(j + 1) + ' epoch' + ' ' + str(i) + ' minibatch' + ':' + str(loss_value)
                print str(j + 1) + ' epoch' + ' ' + str(i) + ' minibatch' + ':' + str(pre)+" "+str(grd)
        save_path = saver.save(sess, model_path)
        print("Model saved in file:%s" % save_path)
        coord.request_stop()
        coord.join(threads)


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
