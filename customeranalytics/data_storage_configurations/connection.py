import pandas as pd
from pathlib import Path

from customeranalytics import Utils, Config
from customeranalytics.utils.paths import Paths


class BaseConnection:
    logs: list[dict] = []
    data_connection: dict = dict()
    data_columns_integration: dict = dict()
    schedule_data: dict = dict()
    actions: dict[dict] = dict()
    ea_configs: dict[dict]
    ml_configs: dict[dict]

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
        self.connection_folder = Path(folder)
        if not self.exists(self.connection_folder, self.connection_file_name):
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

    def get_action(self, data_type):
        return self.default_conf.actions[data_type]

    def get_action_name(self, data_type):
        return self.get_action(data_type)['action_name']

    def remove_action(self, data_type):
        self.connection_conf.actions[data_type] = self.get_action(data_type)
        self.update_connection()

    def update_table(self, table_name: str, values: dict):
        table = getattr(self.connection_conf, table_name)
        for field, value in values.items():
            table[field] = value
        setattr(self.connection_conf, table_name, table)
        self.update_connection()

    def remove_table(self, table_name):
        setattr(self.connection_conf, table_name, getattr(self.default_conf, table_name))

    def collect_data_from_table(self, table='data_connection') -> dict:
        return getattr(self.connection_conf, table)


    def all_connection_keys(self):
        connection_keys = (
            [*self.collect_data_from_table('data_connection').keys()]
            + [*self.collect_data_from_table('data_columns_integration').keys()]
            + [*self.collect_data_from_table('schedule_data').keys()]
        )
        return connection_keys

    def update_ea_ml_config(self, config, config_type):
        setattr(self.connection_conf, config_type, config)
        self.update_connection()


    @staticmethod
    def file_name(data_type):
        return f"{data_type}.parquet"

    @staticmethod
    def insert_update_data_type(self, list_of_obj):
        """
        if data set in list convert to pandas dataframe
        """
        if type(list_of_obj) == list:
            list_of_obj = pd.DataFrame(list_of_obj)
        return list_of_obj

    def get_data_from_folder(self, data_type='orders') -> pd.DataFrame:
        """
        reading parquet file querying connection folder.
        :return: pandas dataframe
        """
        return pd.read_parquet(
            self.connection_folder / self.file_name(data_type)
        )

    def insert_data(self, list_of_obj, data_type):
        (
            self.insert_update_data_type(list_of_obj)
            .to_parquet(
                self.connection_folder / self.file_name(data_type)
            )
        )

    def update_data(self, list_of_obj, data_type):
        """
        reading parquet file querying connection folder.
        concat existed data in the folder upload in to same folder
        bulk insert into the given index.
        :param list_of_obj: list of pandas dataframe
        :param data_type: downloads, orders, etc
        """
        data = pd.concat([
            self.get_data_from_folder(data_type),
            self.insert_update_data_type(list_of_obj, data_type)
        ])
        self.insert_data(data, data_type)


    def check_data_exists(self, data_type):
        """
        Checking data_type is available in the folder
        If the index has not been created, yet, directly send it to the create_index
        :param data_type: downloads, orders, etc
        """
        return self.exists(self.connection_folder, self.file_name(data_type))

