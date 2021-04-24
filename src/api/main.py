import flask
from flask import json, request, jsonify
from werkzeug.exceptions import HTTPException

app = flask.Flask(__name__)
app.config['DEBUG'] = False

@app.errorhandler(Exception)
def handle_all_exceptions(e):
    response = jsonify({
        'message': 'An unhandled exception occured',
        'content': e
    })
    response.status_code = 500
    return response

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'Status': 'OK!',
        'Available Endpoints': [f'/{rule.endpoint}' for rule in app.url_map.iter_rules() if rule.endpoint not in ['home', 'static']]
    })

@app.route('/train', methods=['GET'])
def train():
    raise Exception('test')
    return jsonify({'Status': 'OK!'})

@app.route('/predict', methods=['GET'])
def predict():
    return jsonify({'Status': 'OK!'})

app.run(host='0.0.0.0')
