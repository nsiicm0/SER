import numpy as np
from speechpy.feature import mfcc
from typing import List, Tuple

from ser import Sample
from ser.utils import get_logger, one_hot_encode

_logger = get_logger(__name__)


class FeatureExtractor(object):
    """
        Feature Extractor Base Class

        All feature extractors shall implement this class. 

        Note: The FeatureExtractors shall not one-hot encode the target values. This will be done during training if needed.
    """

    @classmethod
    def extract(cls, samples: List[Sample], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extraction method.
        No need to super call this class, as it would raise an error.
        The extract method should always return an X and a y of the data sample.
        """
        raise NotImplementedError(f'{cls.__name__} has not yet implemented the {cls.__name__}.extract() method!')


class _SpeakerAndGenderAndTextType(FeatureExtractor):
    """
        Basic Speaker, Gender and Text Type Feature Extractor

        The samples have been spoken by either male or female, a multitude of speakers and have a total of 10 different underlying sentences.
    """

    @classmethod
    def extract(cls, samples: List[Sample], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extraction method.

        samples: list
            A list of ser.Samples.
        kwargs: dict
            A dictionary of the following format to enable or disable a certain extraction subset:
            {
                'speaker': True|False,
                'gender': True|False,
                'text': True|False
            } 
            All values default to True!
        """
        X, y = list(), list()
        _speakers = Sample.SPEAKERS
        _genders = list(set(Sample.GENDERS.values()))
        _texts = Sample.TEXTS

        x_speaker, x_gender, x_text = list(), list(), list()

        for sample in samples:
            if 'speaker' not in kwargs or kwargs['speaker']:
                _speaker = sample.get('speaker')
                x_speaker.append(one_hot_encode([_speaker], _speakers)[0])
            if 'gender' not in kwargs or kwargs['gender']:
                _gender = sample.get('gender')
                x_gender.append(one_hot_encode([_gender], _genders)[0])
            if 'text' not in kwargs or kwargs['text']:
                _text = sample.get('text')
                x_text.append(one_hot_encode([_text], _texts)[0])
            y.append(sample.get('emotion'))
        for _features in zip(*filter(lambda _x: len(_x) > 0, [x_speaker, x_gender, x_text])):
            X.append(np.concatenate(_features))
        return X, y


class _MFCC(FeatureExtractor):
    """
        MFCC (Mel-frequency cepstral coefficients) Feature Extractor

        More info here: https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd
        In research, MFCCs have been used quiet extensively to detect emotions in audio.
    """

    @classmethod
    def extract(cls, samples: List[Sample], **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extraction method.

        samples: list
            A list of ser.Samples.
        kwargs: dict
            Arguments for the speechpy.mfcc function.        
        """
        X, y = list(), list()
        for sample in samples:
            if not isinstance(sample, Sample): raise ValueError(f'The element provided is not of type Sample. Got: {type(sample)}')
            audio_buffer = np.frombuffer(sample.get('object').get_array_of_samples(), dtype=np.int16)
            mel_coefficients = mfcc(audio_buffer, sample.get('sample_rate'), **kwargs)
            X.append(np.ravel(mel_coefficients)) # flatten the features
            y.append(sample.get('emotion'))
        return X, y