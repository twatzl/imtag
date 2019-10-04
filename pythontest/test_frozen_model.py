import tensorflow as tf
import os
import numpy as np
import os,glob,cv2
import sys,argparse

# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1]
filename = dir_path +'/' +image_path
image_size=224
num_channels=3
images = []
top_k = 20

##  Reading the image using OpenCV
image = cv2.imread(filename)
 
##   Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)
##   The input to the network is of shape [None image_size image_size num_channels]. 
## Hence we reshape.
 
x_batch = images.reshape(1, image_size,image_size,num_channels)
 
frozen_graph="../data/tensorflowModels/frozen_resnet_v2_152.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
 
with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def,
                          input_map=None,
                          return_elements=None,
                          name=""
      )

## NOW the complete graph with values has been restored
raw_predictions = graph.get_tensor_by_name("resnet_v2_152/predictions/Reshape_1:0")
##raw_predictions = graph.get_tensor_by_name("predictions:0")

probabilities = tf.nn.softmax(raw_predictions)
#predicted_class_index_by_prob = tf.argmax(probabilities, axis=1)

top_k_classes = tf.math.top_k(
    probabilities,
    k=top_k,
    sorted=True)

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("input:0")
y_test_images = np.zeros((1, 2))
sess= tf.Session(graph=graph)
### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch}
result=sess.run(top_k_classes, feed_dict=feed_dict_testing)
print(result)

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

    #print(top_k_classes)
for i in range(0, top_k):
    print("{} {}".format(imagenet_labels[result.indices[0][i]], result.values[0][i]))

