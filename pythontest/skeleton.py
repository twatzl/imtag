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
    return "ok"


if __name__ == '__main__':
    # debug only
    # app.run(debug=True)
    # production
    print(welcome_message)
    serve(app, host=host, port=port)
