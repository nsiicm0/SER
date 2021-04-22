import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return jsonify({'Status': 'OK!'})

app.run(host="0.0.0.0")