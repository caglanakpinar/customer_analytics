import os
from sqlalchemy import create_engine, MetaData
from flask_sqlalchemy import SQLAlchemy
import pandas as pd

from customeranalytics.utils import Utils
from customeranalytics.utils.paths import Paths
from customeranalytics.configs import Config


class SQLiteDB(Paths, Utils, Config):
    def __init__(self):
        self.sqlite_queries = self.read_yaml(self.query_path, "queries.yaml")
        self.engine = create_engine(
            self.sqlite_url,
            connect_args={'check_same_thread': False}
        )
        self.con = self.engine.connect()
        self.db = SQLAlchemy()
        self.tables = self.read_tables()

    @staticmethod
    def query_replacement(query):
        return query.replace("\\", "")

    def insert_query(self, table, columns, values) -> str:
        values = [values[col] for col in columns]
        _query = f"""
        INSERT INTO 
            {table} 
            ({", ".join(columns)})
        VALUES
            ({", ".join([" '{}' ".format(v) for v in values])})
        """
        return self.query_replacement(_query)

    def update_query(self, table, condition, columns, values) -> str:
        values = [
            (col, values[col])
            for col in columns
            if values.get(col) is not None
        ]
        _query = f"""
        UPDATE
            {table}
        SET
            {", ".join([i[0] + " = '" + i[1] + "'" for i in values])}
        WHERE
            {condition}             
        """
        return self.query_replacement(_query)

    def delete_query(self, table, condition) -> str:
        _query = f"""
        DELETE FROM {table}
        WHERE {condition}
        """
        return self.query_replacement(_query)

    def read_query(self, query) -> pd.DataFrame:
        try:
            return pd.read_sql(
                query,
                self.con
            )
        except Exception as e:
            return pd.DataFrame()

    def collect_data_from_table(self, table, query_str=None, return_list=False) -> pd.DataFrame | list[dict]:
        try:
            data =  self.read_query("SELECT * FROM " + table)
            if query_str is not None:
                data = data.query(query_str)
        except Exception as e:
            data = pd.DataFrame()

        if return_list:
            data = data.to_dict('records')
        return data


    def read_tables(self):
        return self.read_query(self.sqlite_queries['tables'])

    def execute_query(self, query):
        self.con.execute(
            query
        )

    def check_for_table_exits(self, table):
        """
        checking sqlite if table is created before. If it not, table is created.
        :params table: checking table name in sqlite
        """
        if table not in list(self.tables['name']):
            self.con.execute(self.sqlite_queries[table])

