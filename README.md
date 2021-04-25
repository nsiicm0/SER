# SER - Speech Emotion Recognition

This repository contains the codebase used for the Use Case Challange as part of the job application for "Machine Learning Engineer (Intern)".

This repository is private due to confidentiality reasons.

## How to use

You can access the API on localhost:5000

### `/train` Endpoint

The `/train` endpoint expects a POST request with a body like this:

```
{
    'data_path': <path to data folder>,             # mandatory, this is where the data will be saved into
    'remote_url': <path to the remote file>,        # mandatory, this is where the data zip resides on the remote host
    'random_state': <int>                           # optional, a random state used for seeding RNGs                         
    'dataset_force': True|False,                    # optional, would force all function calls in dataset
    'dataset_load_split_at': 'min|max|avg|<int>',   # optional, see ser.Dataset.load() for more information
    'dataset_load_normalize': True|False,           # optional, see ser.Dataset.load() for more information
    'dataset_feature_extract_methods': {            # optional, see ser.Dataset.feature_extract() for more information
        'METHOD_NAME': <CONFIG>,                    # METHOD_NAME accepts a valid feature extractor, currently supported: SpeakerAndGenderAndTextType|MFCC
        ...                                         # CONFIG is the config dict as found in ser.Dataset.feature_extract()
    },
    'model_type': <MODEL NAME>,                     # mandatory, currently supported: KerasClassifier|KerasDropoutClassifier
    'model_save_path': <path to model folder>       # mandatory, this is where the model will be saved into (also will hold other training related data)
    'model_config': <MODEL TRAINING CONFIG>         # optional, elsewise default values will be used.
                                                    # Default config: {
                                                    #   'KerasClassifier': {'lr': 0.001, 'epochs': 20, 'batch_size': 32},
                                                    #   'KerasDropoutClassifier': {'lr': 0.001, 'epochs': 100, 'batch_size': 32, 'dropout': 0.2},
                                                    # }
}
```

### `/predict` Endpoint

The `/predict` endpoint expects a POST request with a body like this:

```
{
    model_save_path: <path to model folder>         # mandatory, the path holds the pickles of the model and the training config
    model_id: <model id>                            # mandatory, the ID of the train cycle. (received from the train endpoint response)
    sample_name: <FILENAME>                         # mandatory, the name of the sample to predict. Must be part of the validation set. Is also retrieved from the train endpoint response.

}
```