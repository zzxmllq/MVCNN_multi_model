import tensorflow as tf

def get_loss(input_data,grdtruth):
    l=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=grdtruth,logits=input_data)
    l=tf.reduce_mean(l)
    # tf.add_to_collection('losses',l)
    # return tf.add_n(tf.get_collection('losses'),name='total_loss')
    return l
