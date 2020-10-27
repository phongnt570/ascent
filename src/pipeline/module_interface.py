from configparser import ConfigParser
from typing import List, Set, Tuple


class Module(object):
    def __init__(self, config: ConfigParser):
        self._name = "Module Interface"
        self._config = config

    def run(self, subject_list: List[str], **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self._name
