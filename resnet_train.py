import tensorflow as tf

import model

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('record_path', None, 'Path to training tfrecord file.')
flags.DEFINE_string('resnet50_model_path', None,
                    'Path to pretrained ResNet-50 model.')
flags.DEFINE_string('logdir', None, 'Path to log directory.')
FLAGS = flags.FLAGS


def get_record_dataset(record_path,
                       reader=None, image_shape=[224, 224, 3],
                       num_samples=50000, num_classes=10):
    """Get a tensorflow record file.

    Args:

    """
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1],
                                                                     dtype=tf.int64))}

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=image_shape,
                                              # image_key='image/encoded',
                                              # format_key='image/format',
                                              channels=3),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer between 0 and 9.'}
    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)


def main(_):
    dataset = get_record_dataset(FLAGS.record_path, num_samples=79573,
                                 num_classes=54)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])

    # Data augumentation
    image = tf.image.random_flip_left_right(image)

    inputs, labels = tf.train.batch([image, label],
                                    batch_size=64,
                                    allow_smaller_final_batch=True)

    cls_model = model.Model(is_training=True)
    preprocessed_inputs = cls_model.preprocess(inputs)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    acc = cls_model.accuracy(postprocessed_dict, labels)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.99)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = slim.learning.create_train_op(loss, optimizer,
                                             summarize_gradients=True)

    variables_to_restore = slim.get_variables_to_restore()
    init_fn = slim.assign_from_checkpoint_fn(FLAGS.resnet50_model_path,
                                             variables_to_restore,
                                             ignore_missing_vars=True)

    slim.learning.train(train_op=train_op, logdir=FLAGS.logdir,
                        init_fn=init_fn,
                        save_summaries_secs=20,
                        save_interval_secs=600)


if __name__ == '__main__':
    tf.app.run()
