import tensorflow as tf
import reader
import alexnet
import loss
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 10, 'Number of blocks in each batch')
flags.DEFINE_string('root_path', '/media/zzxmllq/0002605F0001462E/mvcnn/',
                    'Training data file path.')
flags.DEFINE_integer('data_size', 800, 'Number of images')
flags.DEFINE_integer('n_classes', 40, 'Number of classes')
flags.DEFINE_float('keep_prob', 0.5, 'the rate of keep_prob')
flags.DEFINE_integer('max_epoch', 150, 'Number of epoch')
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
tfrecord_filename = FLAGS.root_path + 'train.tfrecords'
# tfrecord_filename = FLAGS.root_path + 'train.tfrecords'
model_path='/media/zzxmllq/0002605F0001462E/mvcnn/model_alexnet/re/alexnet_re.ckpt'

def test():
    filename_queue = tf.train.string_input_producer([tfrecord_filename])
    images, labels = reader.read_and_decode(filename_queue=filename_queue, batch_size=FLAGS.batch_size,shuffle_batch=False)
    # with tf.device('/gpu:0'):
    #     global_step = slim.create_global_step()
    # with tf.device('/cpu:0'):
    #     num_batches_per_epoch = FLAGS.data_size / FLAGS.batch_size
    #     decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    #     lrn_rate = tf.train.exponential_decay(
    #         FLAGS.learning_rate, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
    with tf.device('/gpu:0'):
        # optimizer = tf.train.AdamOptimizer(lrn_rate)
        fc8 = alexnet.model(input_data=images, n_classes=FLAGS.n_classes, keep_prob=FLAGS.keep_prob)
        losses = loss.get_loss(input_data=fc8, grdtruth=labels)
        # train_step = optimizer.minimize(loss=losses, global_step=global_step)
        prediction = alexnet.classify(fc8)
    with tf.device('/cpu:0'):
        saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        total=0.0
        right=0.0
        for i in range(80):
            loss_value, pre,grd = sess.run([losses, prediction,labels])
            print str(i+1)+'image loss :'+str(loss_value)
            print str(i+1) + ' result:' + ':' + str(pre) + " " + str(grd)
            right+=np.sum(np.equal(pre,grd))
            total+=10
            print 'accurcy:'+str(right/total)

        coord.request_stop()
        coord.join(threads)


def main(_):
    test()


if __name__ == '__main__':
    tf.app.run()
