import json

from flask import jsonify
from flask_restful import Api, Resource, reqparse

import ser


def boolean_string(s):
    """
        Argparse type function to handle True and False boolean strings
        Based on: https://stackoverflow.com/a/44561739
    """
    if s is None:
        return False
    s = str(s)
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


class Train(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()

        parser.add_argument('data_path', required=True)
        parser.add_argument('remote_url', required=True)
        parser.add_argument('random_state', type=int, required=False, default=42)
        parser.add_argument('dataset_force', type=boolean_string, required=False, default=False)
        parser.add_argument('dataset_load_split_at', required=False, default=None)
        parser.add_argument('dataset_load_normalize', type=boolean_string, required=False, default=None)
        parser.add_argument('dataset_feature_extract_methods', type=json.loads, required=False, default=None)
        parser.add_argument('model_type', choices=['KerasClassifier', 'KerasDropoutClassifier'], required=True, default='KerasDropoutClassifier')
        parser.add_argument('model_save_path', required=True)
        parser.add_argument('model_config', type=json.loads, required=False, default=None)

        args = parser.parse_args(strict=True)
        trainer = ser.Trainer(args)  # in a productive application, args would need to be validated accordingly in the Resource, however the Trainer handles most of it.
        trainer.prepare_dataset()
        trainer.prepare_model()
        trainer.train()
        trainer.save()

        response = {
            'Model ID': trainer.model_id,
            'Test Samples': trainer.N_test.tolist()
        }
        return response, 200


class Predict(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()

        parser.add_argument('model_save_path', required=True)
        parser.add_argument('model_id', required=True)
        parser.add_argument('sample_name', required=True)

        args = dict(parser.parse_args(strict=True))

        predictor = ser.Predictor(
            model_save_path=args['model_save_path'],
            model_id=args['model_id'],
            sample_name=args['sample_name']
        )
        predictor.restore()
        result = predictor.predict()

        return result, 200