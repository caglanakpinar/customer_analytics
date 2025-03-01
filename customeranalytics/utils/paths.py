import os
from pathlib import Path
import sys, os, inspect


class Paths:
    current_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    base_dir = basedir = os.path.abspath(os.path.dirname(__file__))
    sqlite_url = 'sqlite:///' + os.path.join(base_dir, 'db.sqlite3')

    def abspath_for_sample_data(self):
        """
        get customer_analytics path. Ex: ....../customer_analytics
        :return: current folder path
        """
        current_dir = self.current_dir
        base_name = os.path.basename(self.current_dir)
        while base_name != 'customeranalytics':
            current_dir = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))
            base_name = os.path.basename(current_dir)
        return current_dir