# SER - Speech Emotion Recognition

This repository contains the codebase used for the Use Case Challange as part of the job application for "Machine Learning Engineer (Intern)".

This repository is private due to confidentiality reasons. Due to the extensive amount of time used, this codebase does not use unit tests. Obviously, this is something that would not be done in a production environment.

## How to use

The application is controlled by a Makefile. Simply type `make` to see all available rules. Use `make init` to start the docker container.
The docker container will start with two screens. One running the jupyter server and the other the flask server. You can access them using `screen -r API` or `screen -r JUPYTER` (exit the screens using CTRL+A, D) once attached to the container (`make attach`, detach using CTRL+P, CTRL+Q). The container will forward the ports 5000 (API) and 8888 (JUPYTER).


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
    sample_name: <FILENAME>                         # mandatory, the name of the sample to predict. Must be part of the validation set. It is also retrieved from the train endpoint response.

}
```
