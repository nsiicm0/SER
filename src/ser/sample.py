import os

import pandas as pd

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
    def from_list(cls, samples):
        for sample in samples:
            yield cls(sample)


    SPEAKERS = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']
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
            'object': None,
            'speaker': None,
            'text': None,
            'emotion': None,
            'version': None,
            'features': None
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
            assert self.information['speaker'] in self.SPEAKERS, 'Encountered unknown speaker.'
            self.information['text']  = self.sample_name[2:5]
            assert self.information['text'] in self.TEXTS, 'Encountered unknown text.'
            self.information['emotion']  = self.sample_name[5:6]
            assert self.information['emotion'] in self.EMOTIONS.keys(), 'Encountered unknown emotion.'
            self.information['version']  = self.sample_name[6:7]
            assert self.information['version'] in list('abcdefghijklmnopqrstuvwxyz'), 'Encountered unknown version.'
        except AssertionError:
            _logger.exception(f'There was an issue with the file {self.sample_name}.')


    def to_dict(self, exclude=[]):
        """Returns the sample information dict.

        exclude: list
            Excludes the keys in the list.
        """
        return {k: v for k, v in self.information.items() if k not in exclude}
