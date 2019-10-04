import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# tf.enable_eager_execution()

#path = "../../pics/_DSC0928.JPG"
path = "../../pics/2018-07-14_13-02-50_NIKON D90__DSC0932.JPG"

classifier_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/3"
top_k = 10

m = hub.Module(classifier_url)
height, width = hub.get_expected_image_size(m)

image_file = tf.io.read_file(path)
image_decode = tf.image.decode_jpeg(image_file, channels=3)
image = tf.image.resize(image_decode, [height, width])
image /= 255.0  # normalize to [0,1] range

images = tf.stack([image])  # A batch of images with shape [batch_size, height, width, 3].
raw_predictions = m(images)  # Logits with shape [batch_size, num_classes].

probabilities = tf.nn.softmax(raw_predictions)
predicted_class_index_by_prob = tf.argmax(probabilities, axis=1)

top_k_classes = tf.math.top_k(
    probabilities,
    k=top_k,
    sorted=True)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    pred_class = session.run(predicted_class_index_by_prob)

    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    top_k_classes = session.run(top_k_classes)

    print(top_k_classes)
    for i in range(0, top_k):
        print("{} {}".format(imagenet_labels[top_k_classes.indices[0][i]], top_k_classes.values[0][i]))

# with tf.Session() as sess:
#    f, img = sess.run([probabilities, predicted_class_index_by_prob])
#    print(predicted_class_index_by_prob)


#
# grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
# grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
# grace_hopper = np.array(grace_hopper)/255.0
#
# result = classifier.predict(grace_hopper[np.newaxis, ...])
# predicted_class = np.argmax(result[0], axis=-1)
#
# labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
# imagenet_labels = np.array(open(labels_path).read().splitlines())
#
# predicted_class_name = imagenet_labels[predicted_class]


# with tf.Graph().as_default():
#     test_filename = 'lemon_shark.jpg'
# image = tf.image.decode_jpeg(tf.read_file(path), channels=3)
# processed_image = inception_preprocessing.preprocess_image(image, 224, 224, is_training=False)
# processed_images = tf.expand_dims(processed_image, 0)

# with slim.arg_scope(resnet_v2.resnet_arg_scope()):
#     logits, _ = module(processed_images)
#
#     probabilities = tf.nn.softmax(logits)
#     init_fn = slim.assign_from_checkpoint_fn(
#
#         os.path.join(checkpoints_dir, 'resnet_v2_101.ckpt'),
#
#         slim.get_model_variables('resnet_v2_101'))
#
#     with tf.Session() as sess:
#         init_fn(sess)
#
#     np_image, probabilities = sess.run([image, probabilities])
#
#     probabilities = probabilities[0, 0:]
#
#     sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]
#
#     names = imagenet.create_readable_names_for_imagenet_labels()
