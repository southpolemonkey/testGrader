from flask import Flask
from flask import request
from sid import read_student_id
import requests
from flask import jsonify

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return jsonify(text='Welcome')


@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['file']
    prediction = read_student_id(image)
    return jsonify(result=jsonify(prediction))


if __name__ == 'main':
    app.run(debug=True, port=6500)