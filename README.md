# SER - Speech Emotion Recognition

This repository contains the codebase used for the Use Case Challenge as part of the job application for "Machine Learning Engineer (Intern)".

This repository is private due to confidentiality reasons. 

**Due to the extensive amount of time used for this project, this codebase only uses basic tests. Obviously, this is something that would not be done in a production environment and would follow proper TDD.**

## How to use

The application is controlled using a Makefile. Simply type `make` to see all available rules. Use `make init` to start the docker container.
The docker container will start with two screens. One running the jupyter server and the other the flask server. You can access them using `screen -r API` or `screen -r JUPYTER` (exit the screens using CTRL+A, D) once attached to the container (`make attach`, detach using CTRL+P, CTRL+Q). The container will forward the ports 5000 (API) and 8888 (JUPYTER).

**Note: You will need to enter the JUPYTER screen at least once to get the auth token to access the JUPYTER server.**

### `make`

As mentioned, with make the whole application lifecycle can be controlled.
```
Available rules:

attach              Attach to the running container 
boot-app            Starts the SER application (jupyter and flask api) 
clean               Delete all compiled Python files 
clean-container     Remove the Docker Container 
clean-docker        Remove the Docker Container 
clean-docker-full-only-in-emergency Remove all Docker related data (image and container). Warning: There will be dragons! 
clean-image         Remove the Docker Image 
create-container    Create the Docker Container 
init                Init everything 
init-docker         Initialize the Docker Image 
pip-freeze          Fix all installed python modules to requirements.txt 
pytest              Run Tests 
start-container     Start the Docker Container 
update-env          Updates the environment using using the current requirements.txt 
```

### `/train` Endpoint

Note: The report covers also Random Forest and XGBoost algorithms. They are not implemented in the API endpoints.

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
        'model_type': <MODEL NAME>,                     # mandatory, currently supported: KerasClassifier
        'model_save_path': <path to model folder>       # mandatory, this is where the model will be saved into (also will hold other training related data)
        'model_config': <MODEL TRAINING CONFIG>         # optional, elsewise default values will be used.
                                                        # Default config: {
                                                        #   'KerasClassifier': {'lr': 0.001, 'epochs': 20, 'batch_size': 32, 'dropout': 0.0, 'input_dim': 500},
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

## Remarks

The folder `requests` contains sample requests to be used using the macOS app `rested`.
