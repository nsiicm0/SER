import os
import sys
if os.path.join(os.getcwd(), 'src') not in sys.path:
    # depending on where we launch the app, make sure we can see our code
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import flask
from flask_restful import Api
from flask import json, request, jsonify

from api.resources import Train, Predict

def create_app():
    app = flask.Flask(__name__)
    app.config['DEBUG'] = False

    API = Api(app)
    API.add_resource(Train, '/train')
    API.add_resource(Predict, '/predict')

    @app.errorhandler(Exception)
    def handle_all_exceptions(e):
        response = jsonify({
            'message': 'An unhandled exception occured',
            'content': str(e)
        })
        response.status_code = 500
        return response

    @app.route('/', methods=['GET'])
    def home():
        return jsonify({
            'Status': 'OK!',
            'Available Endpoints': [f'/{rule.endpoint}' for rule in app.url_map.iter_rules() if rule.endpoint not in ['home', 'static']]
        })
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0')
