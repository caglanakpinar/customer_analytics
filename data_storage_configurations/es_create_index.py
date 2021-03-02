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
from dateutil.parser import parse
from sqlalchemy import create_engine, MetaData

from utils import read_yaml, current_date_to_day
from configs import query_path, default_es_port, default_es_host
from os.path import abspath, join
from data_storage_configurations.query_es import QueryES
from data_storage_configurations.data_access import GetData
from data_storage_configurations import create_data_access_parameters

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
        self.query_es = QueryES()
        self.es_cons = pd.DataFrame()
        self.schedule = pd.DataFrame()
        self.start_date = None
        self.ebd_date = current_date_to_day()
        self.port = default_es_port
        self.host = default_es_host
        self.latest_session_transaction_date = None

    def insert_query(self, table, columns, values):
        values = [values[col] for col in columns]
        _query = "INSERT INTO " + table + " "
        _query += " (" + ", ".join(columns) + ") "
        _query += " VALUES (" + ", ".join([" '{}' ".format(v) for v in values]) + ") "
        _query = _query.replace("\\", "")
        return _query

    def update_query(self, table, condition, columns, values):
        values = [(col, values[col]) for col in columns if values.get(col, None) is not None]
        _query = "UPDATE " + table
        _query += " SET " + ", ".join([i[0] + " = '" + i[1] + "'" for i in values])
        _query +=" WHERE " + condition
        _query = _query.replace("\\", "")
        print(_query)
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

    def update_schedule(self, date):
        self.check_for_table_exits(table='schedule_data')
        try:
            con.execute(self.update_query(table='schedule_data',
                                          condition=" tag = '" + self.es_tag + "' ",
                                          columns=['max_date_of_order_data'],
                                          values= {'max_date_of_order_data': date}
                                          ))
        except Exception as e:
            print(e)

    def collect_es_connection_infos(self):
        self.es_cons = pd.read_sql("SELECT * FROM es_connection WHERE tag = '" + self.es_tag + "' ", con)
        self.port = list(self.es_cons['port'])[0]
        self.host = list(self.es_cons['host'])[0]

    def create_index_connection(self):
        self.collect_es_connection_infos()
        self.query_es = QueryES(port=self.port, host=self.port)

    def check_and_create_index(self, index):
        self.create_index_connection()
        accept = self.query_es.check_index_exists(index=index)
        return accept

    def get_schedule_data(self):
        self.schedule = pd.read_sql("SELECT * FROM schedule_data WHERE tag = '" + self.es_tag + "' ", con)

    def get_start_date(self):
        self.start_date = list(self.schedule['max_date_of_order_data'])[0]

    def create_index_obj(self, configs, index_type='index'):
        """

        """



    def change_columns_format(self, data, data_source_type, columns):
        columns = list(data.columns)
        if data_source_type == 'connection':
            data['session_start_date'] = data['session_start_date'].apply(lambda x: parse(x))
            data['payment_amount'] = data['payment_amount'].apply(lambda x: float(x))
            if 'discount_amount' in ['columns']:
                data['discount_amount'] = data['discount_amount'].apply(lambda x: float(x) if x == x else None)
            if 'date' in ['columns']:
                data['date'] = data['date'].apply(lambda x: parse(x) if x == x else None)

        if data_source_type == 'products':
            data['date'] = data['date'].apply(lambda x: parse(x) if x == x else None)




    def get_data(self, conf, data_source_type):
        """
        {'data_source': connection[index + '_data_source_type'],
            'date': date,
            'data_query_path': connection[index + '_data_query_path'],
            'test': test,
            'config': {'host': connection[index + '_host'],
                       'port': connection[index + '_port'],
                       'password': connection[index + '_password'],
                       'user': connection[index + '_user'], 'db': connection[index + '_db']}
            }
        """
        gd = GetData(data_source=conf['data_source'],
                     data_query_path=conf['data_query_path'], config=conf['config'], test=1000)
        gd.query_data_source()
        gd = GetData(data_source=conf['data_source'],
                     data_query_path=conf['data_query_path'], config=conf['config'])
        gd.query_data_source()
        return gd.data.rename(columns={conf['columns'][col]:col for col in conf['columns']})


    def merge_data_sets(self, configs, type, index):
        _data_sets = {}
        for ds_type in configs:
            _data_sets['ds_type'] = self.get_data(configs[ds_type], ds_type)

    def execute_index(self):
        """
            data_config = {'orders': {'main':
                                         {'connection': {},
                                          'action': [],
                                          'product': [],
                                          'promotion': []
                                          },
                                      dimensions': [{'connection': {},
                                                     'action': [],
                                                     'product': [],
                                                     'promotion': []
                                                     }
                                                    ]
                          },
               'downloads': {'main': {'connection': {},
                                      'action': []
                                      },
                             'dimensions': [{'connection': {},
                                             'action': []
                                             }
                                            ]
                             }
               }
        """
        success = True
        self.get_schedule_data()
        try:
            for _data_type in self.data_connection_structure:
                for _type in _data_type:
                    if _type == 'main':
                        _index = 'orders'
                        

            if self.check_and_create_index(_index):
                self.get_start_date()






            last_schedule_triggered_date = str(current_date_to_day())
            self.update_schedule(date=last_schedule_triggered_date)
            self.logs_update(logs={"page": "data-execute",
                                   "info": "indexes on " + self.es_tag + " are created safely",
                                   "color": "green"})
        except Exception as e:
            success = False
            self.logs_update(logs={"page": "data-execute",
                                   "info": "indexes on " + self.es_tag + " are created failed",
                                   "color": "red"})


