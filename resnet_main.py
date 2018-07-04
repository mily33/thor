import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
import tensorflow.contrib.slim as slim
from baselines.a2c import *
# slim = tf.contrib.slim


def processing(image):
    image = cv2.resize(image, (299, 299))
    image = np.array([image], dtype=np.float32)
    image = image / 255
    return image


def resnet_fm(input_ph):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, endpoints = resnet_v2.resnet_v2_50(
            input_ph, num_classes=None,
            is_training=False, reuse=tf.AUTO_REUSE)
        feature_map = tf.squeeze(net, axis=[1, 2])
    return feature_map


def model(state_ph, target_ph, ac_dim, scope):
    state_fm = resnet_fm(state_ph)
    target_fm = resnet_fm(target_ph)

    with tf.variable_scope(scope, reuse=False):
        current_fc = slim.fully_connected(state_fm, num_outputs=512, activation_fn=None, scope='current_fc')
        target_fc = slim.fully_connected(target_fm, num_outputs=512, activation_fn=None, scope='target_fc')

        feature_map = tf.concat([current_fc, target_fc], 1)

        embedding_fusion = slim.stack(feature_map, slim.fully_connected,
                                      [512, 512], scope='embedding_fusion')

        value = slim.fully_connected(embedding_fusion, 1, activation_fn=None, scope='value')
        logit = slim.fully_connected(embedding_fusion, ac_dim, activation_fn=None, scope='policy')

    return value, logit


def main():
    learning_rate = 0.01
    alpha = 0.5
    ac_dim = 4
    gamma = 0.99
    tf.set_random_seed(1)
    current_state = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
    next_state = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
    target = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
    reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    done_ph = tf.placeholder(shape=[None], dtype=tf.float32)

    current_value, logit = model(current_state, target, ac_dim=4, scope='current_state')
    next_value, _ = model(next_state, target, ac_dim=4, scope='next_state')

    # =========== update actor ===========
    sy_ac_na = tf.placeholder(shape=[None], name='ac', dtype='int32')
    sy_adv_n = tf.placeholder(shape=[None], name='adv', dtype='float32')

    sy_sample_na = tf.reshape(tf.multinomial(logit, 1), [-1])
    sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=logit)

    loss_actor = -tf.reduce_mean(sy_logprob_n * sy_adv_n)

    # =========== update critic ===========
    q_var = current_value * tf.one_hot(sy_ac_na, ac_dim)
    target_q_var = reward + (1 - done_ph) * next_value * gamma

    loss_critic = tf.reduce_mean(tf.squared_difference(target_q_var, q_var))
    loss = alpha * loss_actor + (1 - alpha) * loss_critic
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current_state')
    update_op1 = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list)

    model_path = '/home/mily/PycharmProjects/models/resnet_v2/resnet_v2_50.ckpt'
    init_fn = slim.assign_from_checkpoint_fn(model_path, slim.get_model_variables(), ignore_missing_vars=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    init_fn(sess)

    img = cv2.imread('img.jpg')
    target_img = cv2.imread('img2.jpg')
    img = processing(img)
    target_img = processing(target_img)

    value, policy = sess.run([current_value, logit], feed_dict={current_state: img, target: target_img})
    print(value, policy)

    writer = tf.summary.FileWriter('./log/')
    writer.add_graph(sess.graph)


if __name__ == '__main__':
    main()
