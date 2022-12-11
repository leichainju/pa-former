""" loading examples from file
"""

import os.path

import jsonlines
from alive_progress import alive_bar

from ..build import PIPELINES
from .utils import count_file_lines


@PIPELINES.register_module()
class LoadExamplesFromJsonl:
    def __init__(self, config):
        self.train_file = config.get('train_file')
        self.test_file = config.test_file

    def __call__(self, pipelines, logger):
        train_exs = None
        if self.train_file is not None:
            train_exs = self._load_data(self.train_file, pipelines)
            logger.info(f'Num train examples = {len(train_exs)}')

        dev_exs = self._load_data(self.test_file, pipelines)
        logger.info(f'Num dev examples = {len(dev_exs)}')

        return train_exs, dev_exs

    @staticmethod
    def _load_data(file_path, pipelines):
        """Load examples from preprocessed file. One example per line, JSON encoded."""
        num_exs = count_file_lines(file_path)

        examples = []
        with alive_bar(num_exs, bar='classic', title=os.path.basename(file_path)) as bar:
            with jsonlines.open(file_path, 'r') as reader:
                for raw_ex in reader:
                    ex = pipelines(raw_ex)
                    if ex is not None:
                        examples.append(ex)
                    bar()

        return examples
