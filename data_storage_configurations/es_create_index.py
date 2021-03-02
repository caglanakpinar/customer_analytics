import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import pandas as pd
import datetime
import random
from time import gmtime, strftime
import pytz
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import argparse
from sqlalchemy import create_engine, MetaData

from utils import read_yaml
from configs import query_path
from os.path import abspath, join

engine = create_engine('sqlite://///' + join(abspath(""), "web", 'db.sqlite3'), convert_unicode=True, connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


class CreateIndex:
    """
    This class is converting input data to elasticsearch index.
    This allow us to query data from elasticsearch.
    Each business has on its own way to store data.
    This is a generic way to store the data into the elasticsearch indexes
    """
    def __init__(self, es_tag, data_connection_structure):
        """
        data_sets:
            *   orders:
                each purchased orders row by row
                columns: order_id, client, start_date, purchase_date, purchase_amount, discount_amount
                optional columns: purchase_date, discount_amount
            *   products:
                ****** OPTIONAL DATA *******
                each purchased orders of products row by row.
                columns: order_id, product_id, product_category, product_price, rank
                optional columns: product_category, product_price, rank
            *   promos:
                ****** OPTIONAL DATA *******
                each order of promotion status
                please assign only if there is discount at order
                columns: order_id, promotion_id
            *   sessions:
                columns: session_id, client_id
            *   actions:
                ****** OPTIONAL DATA *******
                columns: session_id, client_id, action_1 (boolean: True/False), .. action_10 (boolean: True/False)
            *   action_details:
                ****** OPTIONAL DATA *******
                detail of action
                columns: session_id, action_name (action_1, action_2, ..., action_10), detail_1, detail_2, detail_3


        indicator columns:
        These are type of indicator columns;
        e.g.
        :param tag:
        """
        self.es_tag = es_tag
        self.data_connection_structure = data_connection_structure
        self.sqlite_queries = read_yaml(query_path, "queries.yaml")
        self.tables = pd.read_sql(self.sqlite_queries['tables'], con)

    def insert_query(self, table, columns, values):
        values = [values[col] for col in columns]
        _query = "INSERT INTO " + table + " "
        _query += " (" + ", ".join(columns) + ") "
        _query += " VALUES (" + ", ".join([" '{}' ".format(v) for v in values]) + ") "
        _query = _query.replace("\\", "")
        return _query

    def check_for_table_exits(self, table, query=None):
        if table not in list(self.tables['name']):
            if query is None:
                con.execute(self.sqlite_queries[table])
            else:
                con.execute(query)

    def logs_update(self, logs):
        self.check_for_table_exits(table='logs')
        try:
            con.execute(self.insert_query(table='logs',
                                          columns=self.sqlite_queries['columns']['logs'][1:],
                                          values=logs
                                          ))
        except Exception as e:
            print(e)

    def execute_index(self):
        """

        """
        print("HELLLL YEEAAAAAAA!!!!")
        self.logs_update(logs={"page": "data-execute", "info": "indexes on " + self.es_tag + " are created safely", "color": "green"})
