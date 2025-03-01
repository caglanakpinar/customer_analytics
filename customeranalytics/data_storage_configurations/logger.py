import logging
from os.path import join

from customeranalytics import Paths


class LogsBasicConfeger(Paths):
    def __init__(self):
        file_path = join(self.current_dir, "logs.log")
        logging.basicConfig(
            filename=file_path,
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )



