import tensorflow as tf
import tensorflow_hub as hub
from waitress import serve
from flask import Flask
from flask import request

app = Flask(__name__)

host = '0.0.0.0'
port = 5000
classifier_name = "classifier_skeleton"
welcome_message = classifier_name + " running on server " + host + ":" + str(port)


@app.route('/')
def index():
    return welcome_message


@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    if data is None:
        return "Error: mimetype must be application/json !"

    print(data)
    print(data['hello'])
    classifier()
    return "ok"


def classifier():
    tf.enable_eager_execution()

    module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
    embed = hub.KerasLayer(module_url)
    embeddings = embed(["A long sentence.", "single-word",
                        "http://example.com"])
    print(embeddings.shape)  #(3,128)


if __name__ == '__main__':
    # debug only
    # app.run(debug=True)
    # production
    print(welcome_message)
    path = "/home/twatzl/go/src/imtag/pics/DSC_9949_small.jpg"
    image_file = tf.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_file, channels=3)
    with tf.Session() as sess:
        f, img = sess.run([image_file, image_decoded])
        print(f)
        print(img)

    serve(app, host=host, port=port)
