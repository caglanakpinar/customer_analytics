import os
from pathlib import Path
import sys, os, inspect


class Paths:
    current_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    base_dir = os.path.abspath(os.path.dirname(__file__))
    query_path = os.path.join(parent_dir, "docs")
    connection_folder = Path()
    connection_file_name = "connection.yaml"

    @staticmethod
    def exists(folder: Path, file_name):
        return (folder / Path(file_name)).exists()

    @staticmethod
    def folder_file_list(folder: Path):
        return list(folder.iterdir())

    def abspath_for_sample_data(self):
        """
        get customer_analytics path. Ex: ....../customer_analytics
        :return: current folder path
        """
        current_dir = self.parent_dir
        base_name = os.path.basename(self.parent_dir)
        while base_name != 'customeranalytics':
            current_dir = os.path.dirname(
                os.path.abspath(inspect.getfile(inspect.currentframe())))
            base_name = os.path.basename(current_dir)
        return current_dir