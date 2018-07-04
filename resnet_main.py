import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
import tensorflow.contrib.slim as slim


def processing(image):
    image = cv2.resize(image, (299, 299))
    image = np.array([image], dtype=np.float32)
    image = image / 255
    return image


def main():
    size_feature_map = 512

    preprocessed_inputs = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
    target = tf.placeholder(shape=[None, 299, 299, 3], dtype=tf.float32)
    embedding_fusion_input = tf.placeholder(shape=[None, size_feature_map * 2], dtype=tf.float32)
    model_path = '/home/mily/PycharmProjects/models/resnet_v2/resnet_v2_50.ckpt'

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, endpoints = resnet_v2.resnet_v2_50(
            preprocessed_inputs, num_classes=None,
            is_training=False)
    net = tf.squeeze(net, axis=[1, 2])
    feature_map = slim.fully_connected(net, num_outputs=size_feature_map,
                                           activation_fn=None, scope='fc')
    embedding_fusion = slim.stack(embedding_fusion_input, slim.fully_connected,
                                  [size_feature_map, size_feature_map], scope='embedding_fusion')
    value = slim.fully_connected(embedding_fusion, 1, activation_fn=None, scope='value')
    policy = slim.fully_connected(embedding_fusion, 4, activation_fn=tf.nn.softmax, scope='policy')

    init_fn = slim.assign_from_checkpoint_fn(model_path, slim.get_model_variables(), ignore_missing_vars=True)
    sess = tf.Session()
    sess.__init__()
    sess.run(tf.global_variables_initializer())
    init_fn(sess)

    img = cv2.imread('img.jpg')
    target = cv2.imread('img2.jpg')
    img = processing(img)
    target = processing(target)

    current_state_feature_map = sess.run(feature_map, feed_dict={preprocessed_inputs: img})
    target_state_feature_map = sess.run(feature_map, feed_dict={preprocessed_inputs: target})
    feature_map = np.concatenate([current_state_feature_map, target_state_feature_map], 1)
    fusion = sess.run(embedding_fusion, feed_dict={embedding_fusion_input: feature_map})



if __name__ == '__main__':
    main()