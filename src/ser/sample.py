import os
from copy import deepcopy

import pandas as pd
from pydub import AudioSegment
from pydub.utils import db_to_float, ratio_to_db

from ser.utils import get_logger

_logger = get_logger(__name__)


class Sample(object):
    """
        Sample class

        This wrapper class is designed to mimic an interface to a sample.
        
        sample_path: str
            The path to the sample.
    """

    @classmethod
    def from_list(cls, samples: list) -> list:
        """Creates multiple samples from a list of paths.
        
        samples: list
            List of paths to samples

        Yields Sample objects        
        """
        for sample in samples:
            yield cls(sample)

    @classmethod
    def from_dict(cls, information_dict: dict) -> 'Sample':
        """Instantiates a new Sample object from a provided information dictionary.

        information_dict: dict
            Needs to be a valid information dictionary as seen in Sample.__init__()

        Returns a Sample object
        """
        if 'path' not in information_dict:
            raise ValueError(f'The provided information_dict does not seem to be a valid dict. Got: {information_dict}')
        obj = cls(information_dict['path'])
        for k in obj.information:
            if k not in information_dict:
                raise ValueError(f'The provided information_dict does not seem to be a valid dict. Got: {information_dict}')
            obj.information[k] = information_dict[k]
        return obj


    SPEAKERS = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']
    GENDERS = {'03': 'Male', '08': 'Female', '09': 'Female', '10': 'Male', '11': 'Male', '12': 'Male', '13': 'Female', '14': 'Female', '15': 'Male', '16': 'Female'}
    TEXTS = ['a01', 'a02', 'a04', 'a05', 'a07', 'b01', 'b02', 'b03', 'b09', 'b10']
    EMOTIONS = {
        'W': 'Ärger (Wut)',
        'L': 'Langeweile',
        'E': 'Ekel',
        'A': 'Angst',
        'F': 'Freude',
        'T': 'Trauer',
        'N': 'Neutral'
    }


    def __init__(self, sample_path: str):
        self.sample_path = sample_path
        self.sample_name = os.path.basename(self.sample_path)
        _logger.debug(f'Instantiating Sample: {self.sample_name}')
        if os.path.splitext(self.sample_name)[1].lower() != '.wav':
            _logger.warning(f'Sample at {self.sample_path} does not seem to be a wav file!')
        
        self.information = dict({
            'name': self.sample_name,
            'path': self.sample_path,
            'length': None,
            'sample_rate': None,
            'object': None,
            'speaker': None,
            'gender': None,
            'text': None,
            'emotion': None,
            'version': None,
            'features': None,
            'chunk': 1
        })

        self._parse()


    def __repr__(self):
        chunks = list()
        for k, v in self.information.items():
            if k not in ['object', 'features']: chunks.append((k, v))
        representation = ', '.join([f'{chunk[0]}={chunk[1]}' for chunk in chunks])
        return f'<Sample {representation}>'


    def _parse(self) -> None:
        try:
            assert len(self.sample_name) == 11, 'Encountered unknown filename.'
            self.information['speaker'] = self.sample_name[0:2]
            self.information['gender'] = self.GENDERS[self.information['speaker']]
            assert self.information['speaker'] in self.SPEAKERS, 'Encountered unknown speaker.'
            self.information['text']  = self.sample_name[2:5]
            assert self.information['text'] in self.TEXTS, 'Encountered unknown text.'
            self.information['emotion']  = self.sample_name[5:6]
            assert self.information['emotion'] in self.EMOTIONS.keys(), 'Encountered unknown emotion.'
            self.information['version']  = self.sample_name[6:7]
            assert self.information['version'] in list('abcdefghijklmnopqrstuvwxyz'), 'Encountered unknown version.'
        except AssertionError:
            _logger.exception(f'There was an issue with the file {self.sample_name}.')


    def to_dict(self, exclude=[]) -> dict:
        """Returns the sample information dict.

        exclude: list
            Excludes the keys in the list.
        """
        return {k: v for k, v in self.information.items() if k not in exclude}


    def get(self, attribute: str) -> object:
        """Returns an attribute from the information dict.
        
        attribute: str
            The key in the information dict.
        """
        if attribute not in self.information:
            raise ValueError(f'Invalid attribute provided. Got: {attribute}. Expected on of {list(self.information.keys())}')
        return self.information[attribute]


    def load_from_disk(self) -> 'Sample':
        """Loads the sample from disk.

        Returns a pydub.AudioSegment object        
        """
        audio = AudioSegment.from_wav(self.get('path'))
        self.information['sample_rate'] = audio.frame_rate
        self.information['length'] = len(audio)
        self.information['object'] = audio
        return self

    def split_and_pad(self, length: int) -> list:
        """Splits and pads this sample into sample(s) of length

        length: int
            The length of the new sample(s)
        
        Returns new Sample objects of length
        """
        sample = self.get('object')
        sample_length = self.get('length')
        delta = length - sample_length
        chunks = list()
        if delta < 0: # sample needs splitting
            chunks.extend(sample[::length])
        else:
            chunks.append(sample)
        # pad the last chunk
        last_chunk = chunks[-1]
        to_pad = length - len(last_chunk)
        padding = AudioSegment.silent(duration=to_pad, frame_rate=last_chunk.frame_rate)
        # combine the samples and also make sure we really do not exceed the length due to rounding issues
        chunks[-1] = last_chunk.append(padding, crossfade=0)[:length]
        # create new samples
        samples = list()
        for i, chunk in enumerate(chunks):
            _info_dict = deepcopy(self.information)
            _info_dict['object'] = chunk
            _info_dict['length'] = len(chunk)
            _info_dict['chunk'] += i
            samples.append(Sample.from_dict(_info_dict))

        return samples

    
    def normalize(self, max_value: int, headroom: float=0.1) -> None:
        """Normalize current sample to a max_value.

        Based on implementation of pydub: https://github.com/jiaaro/pydub/blob/master/pydub/effects.py#L36-L49

        max_value: int
            The value to normalize onto.

        headroom: float
            headroom is how close to the maximum volume to boost the signal up to (specified in dB)
        """
        sample = self.get('object')
        target_peak = sample.max_possible_amplitude * db_to_float(-headroom)
        needed_boost = ratio_to_db(target_peak / max_value)
        self.information['object'] = sample.apply_gain(needed_boost)
