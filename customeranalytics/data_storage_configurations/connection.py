import pandas as pd

from customeranalytics import Utils, Config
from customeranalytics.utils.paths import Paths


class BaseConnection:
    logs: list[dict] = []
    data_connection: dict = dict()
    data_columns_integration: dict = dict()
    schedule_data: dict = dict()
    actions: dict[dict] = dict()

    @classmethod
    def connection_config(cls, arguments: dict):
        _cls = BaseConnection()
        for a, v in arguments.items():
            setattr(_cls, a, v)
        return _cls


class Connection(Paths, Utils, Config):
    def __init__(self):
        self.default_conf = BaseConnection.connection_config(
            self.read_yaml(self.query_path, self.connection_file_name)
        )
        self.connection_conf = BaseConnection.connection_config(
            self.read_yaml(self.query_path, self.connection_file_name)
        )

    def update_connection(self):
        self.write_yaml(
            self.connection_conf,
            self.connection_folder,
            self.connection_file_name,
        )

    def create_configuration(self, folder: str):
        self.connection_folder = folder
        if not self.exists(folder, self.connection_file_name):
            self.update_connection()

    def check_for_table_exits(self, table: str):
        """
        checking if connection is created at connection.yanl in given folder
        :params table: checking table name in sqlite
        """
        conn = getattr(self.connection_conf, table)
        for i, v in conn.items():
            if v is None:
                return False
        return True

    def add_logs(self, log: dict):
        self.connection_conf.logs.append(log)
        self.update_connection()

    def update_actions(self, action: dict):
        self.connection_conf.actions[action['data_type']] = action
        self.update_connection()

    def remove_action(self, data_type):
        self.connection_conf.actions[data_type] = self.default_conf.actions[data_type]
        self.update_connection()

    def collect_data_from_table(self, table='data_connection') -> dict:
        return getattr(self.connection_conf, table)


    def all_connection_keys(self):
        connection_keys = (
            [*self.collect_data_from_table('data_connection').keys()]
            + [*self.collect_data_from_table('data_columns_integration').keys()]
            + [*self.collect_data_from_table('schedule_data').keys()]
        )
        return connection_keys


