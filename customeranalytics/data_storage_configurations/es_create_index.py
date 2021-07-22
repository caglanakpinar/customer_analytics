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
from flask_login import current_user
from os.path import abspath, join

from customeranalytics.utils import read_yaml, current_date_to_day, convert_to_date, convert_to_iso_format, formating_numbers
from customeranalytics.configs import query_path, default_es_port, default_es_host, none_types
from customeranalytics.configs import orders_index_columns, downloads_index_columns, not_required_columns, not_required_default_values
from customeranalytics.data_storage_configurations.query_es import QueryES
from customeranalytics.data_storage_configurations.data_access import GetData

try:
    engine = create_engine('sqlite://///' + join(abspath(""), "web", 'db.sqlite3'), convert_unicode=True, connect_args={'check_same_thread': False})
    metadata = MetaData(bind=engine)
    con = engine.connect()
except Exception as e:
    engine = create_engine('sqlite://///' + join(parentdir, "web", 'db.sqlite3'), convert_unicode=True, connect_args={'check_same_thread': False})
    metadata = MetaData(bind=engine)
    con = engine.connect()


class CreateIndex:
    """
    This class is converting input data to elasticsearch index.
    This allow us to query data from elasticsearch.
    Each business has on its own way to store data.
    This is a generic way to store the data into the elasticsearch indexes
    """
    def __init__(self, data_connection_structure, data_columns, actions):
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
        :param data_connection_structure: data connections
        :param data_columns: matched columns
        """
        self.data_connection_structure = data_connection_structure
        self.data_columns = data_columns
        self.actions = actions
        self.sqlite_queries = read_yaml(query_path, "queries.yaml")
        self.tables = pd.read_sql(self.sqlite_queries['tables'], con)
        self.query_es = QueryES()
        self.es_cons = pd.DataFrame()
        self.schedule = pd.DataFrame()
        self.start_date, self.end_date = None, None
        self.ebd_date = current_date_to_day()
        self.port = default_es_port
        self.host = default_es_host
        self.latest_session_transaction_date = None
        self.result_data = pd.DataFrame()
        self.job_start_date = None
        self.info_logs_for_chat = lambda info: {'user': 'info',
                                                'date': str(current_date_to_day())[0:19],
                                                'user_logo': 'info.jpeg',
                                                'chat_type': 'info', 'chart': '',
                                                'general_message': info, 'message': ''}

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
        return _query

    def check_for_table_exits(self, table, query=None):
        try:
            if table not in list(self.tables['name']):
                if query is None:
                    con.execute(self.sqlite_queries[table])
                else:
                    con.execute(query)
        except Exception as e:
            print(e)

    def logs_update(self, logs):
        """
        logs table in sqlite table is updated.
        chats table in sqlite table is updated.
        """
        try: self.check_for_table_exits(table='logs')
        except Exception as e: print(e)

        try: self.check_for_table_exits(table='chat')
        except Exception as e: print(e)

        try:
            logs['login_user'] = current_user
            logs['log_time'] = str(current_date_to_day())[0:19]
            logs['general_message'] = logs['info']
            con.execute(self.insert_query(table='logs',
                                          columns=self.sqlite_queries['columns']['logs'][1:],
                                          values=logs
                                          ))
        except Exception as e:
            print(e)

        try: con.execute(self.insert_query(table='chat', columns=self.sqlite_queries['columns']['chat'][1:],
                                           values=self.info_logs_for_chat(logs['info'])))
        except Exception as e: print(e)

    def update_schedule(self, date):
        self.check_for_table_exits(table='schedule_data')
        try:
            con.execute(self.update_query(table='schedule_data',
                                          condition=" id = 1",
                                          columns=['max_date_of_order_data'],
                                          values= {'max_date_of_order_data': date}
                                          ))
        except Exception as e:
            print(e)

    def collect_es_connection_infos(self):
        self.es_cons = pd.read_sql("SELECT * FROM es_connection", con)
        self.port = list(self.es_cons['port'])[0]
        self.host = list(self.es_cons['host'])[0]

    def create_index_connection(self):
        self.collect_es_connection_infos()
        self.query_es = QueryES(port=self.port, host=self.host)

    def index_count(self, index):
        _count = int(self.query_es.es.cat.count(index, params={"format": "json"})[0]['count'])
        return _count

    def check_and_create_index(self, index):
        self.create_index_connection()
        accept = False if self.query_es.check_index_exists(index=index) is None else True
        if accept:
            if self.index_count(index) != 0:
                accept = True
        return accept

    def get_schedule_data(self):
        self.schedule = pd.read_sql("SELECT * FROM schedule_data", con)

    def get_start_date(self):
        if list(self.schedule['time_period'])[0] != 'once':
            self.start_date = convert_to_date(list(self.schedule['max_date_of_order_data'])[0])

    def get_end_date(self):
        if list(self.schedule['time_period'])[0] != 'once':
            self.end_date = current_date_to_day()

    def design_order_basket(self, x):
        try:
            result = {}
            for i in x:
                for p in list(i.keys()):
                    result[p] = i[p]
            return result
        except Exception as e:
            print(e)
            return None

    def change_columns_format(self, data, data_source_type):
        columns = list(data.columns)
        if data_source_type in ['orders', 'products']:
            data['order_id'] = data['order_id'].apply(lambda x: str(x))
        if data_source_type in ['orders', 'downloads']:
            data['client'] = data['client'].apply(lambda x: str(x))
        if data_source_type == 'orders':
            data['session_start_date'] = data['session_start_date'].apply(lambda x: parse(x))
            data['payment_amount'] = data['payment_amount'].apply(
                lambda x: float(x) if x == x and x not in ['None', None, 'nan'] else None)
            if 'discount_amount' in columns:
                data['discount_amount'] = data['discount_amount'].apply(
                    lambda x: float(x) if x == x and x not in ['None', None, 'nan'] else None)
            if 'date' in columns:
                data['date'] = data['date'].apply(
                    lambda x: parse(x) if x == x and x not in ['None', None, 'nan'] else None)
            if 'promotion_id' in columns:
                data['promotion_id'] = data['promotion_id'].apply(
                    lambda x: x if x == x and x not in ['None', None, 'nan'] else None)
        if data_source_type == 'downloads':
            data['download_date'] = data['download_date'].apply(
                lambda x: parse(x) if x == x and x not in ['None', None, 'nan'] else None)
            if 'signup_date' in columns:
                data['signup_date'] = data['signup_date'].apply(
                    lambda x: parse(x) if x == x and x not in ['None', None, 'nan'] else None)
        if data_source_type == 'products':
            if 'price' in columns:
                data['price'] = data['price'].apply(
                    lambda x: float(x) if x == x and x not in ['None', None, 'nan'] else None)
        return data

    def get_index_of_last_date(self, date_column, index):
        try:
            match = {"size": 1, "from": 0, "_source": True, "sort": {date_column: "desc"}, }
            res = self.query_es.es.search(index=index, body=match)['hits']['hits']
            return convert_to_date([r['_source'][date_column] for r in res][0])
        except Exception as e:
            print("session_start_date/download_date is not inserted into the indexes.")
            return None

    def collect_prev_downloads(self):
        try:
            match = {"size": 100000, "from": 0, "_source": True}
            res = self.query_es.es.search(index='downloads', body=match)['hits']['hits']
            return [r['_source']['client'] for r in res]
        except Exception as e:
            print(e)
            return []

    def filter_dates(self, data, data_source_type):
        """

        """
        try:
            if data_source_type in ['orders', 'downloads']:
                date_column = "session_start_date" if data_source_type == 'orders' else "download_date"
                max_index_date = self.get_index_of_last_date(date_column, data_source_type)
                if self.start_date is not None:
                    data = data[data[date_column] >= self.start_date]
                if self.end_date is not None:
                    data = data[data[date_column] < self.end_date]
                if max_index_date is not None:
                    data = data[data[date_column] >= max_index_date]
                if data_source_type == 'downloads':
                    data[~data['client'].isin(self.collect_prev_downloads())]
        except Exception as e:
            print(e)
        return data

    def match_data_columns(self, data):
        cols = list(data.columns)
        renaming = {}
        for col in self.data_columns:
            if self.data_columns[col] in cols:
                renaming[self.data_columns[col]] = col if col != 'client_2' else 'client'
        return data.rename(columns=renaming)

    def check_for_not_required_columns(self, data, data_source_type):
        _columns = list(data.columns)
        for col in not_required_columns[data_source_type]:
            if col not in _columns:
                data[col] = not_required_default_values[col]
        return data

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
        data = self.match_data_columns(data=gd.data)
        data = self.change_columns_format(data, data_source_type)
        data = self.filter_dates(data, data_source_type)
        data = self.check_for_not_required_columns(data, data_source_type)
        return data

    def merge_orders(self, order_source, product_source):
        orders = pd.DataFrame()
        try:
            orders = self.get_data(conf=order_source, data_source_type='orders')
        except Exception as e:
            print(e)

        if len(orders) != 0:
            try:
                if product_source.get('data_query_path', None) is not None:
                    products = self.get_data(conf=product_source, data_source_type='products')
                    products['basket'] = products.apply(
                        lambda row: {row['product']: {'price': row['price'],
                                                      'category': row['category']}}, axis=1)
                    products = products.groupby("order_id").agg({"basket":
                                                         lambda x: self.design_order_basket(x)}).reset_index()
                    products['total_products'] = products['basket'].apply(lambda x: len(x.keys()))
                    orders = pd.merge(orders, products, on='order_id', how='left')
                    orders['basket'] = orders['basket'].fillna({})
                    del products
                else: orders['basket'] = None
            except Exception as e:
                print(e)
                orders['basket'] = None
        return orders

    def insert_to_index(self, data, index):
        """
        Sessions (Orders) Index Document;
            {'_index': 'orders',
             '_type': '_doc',
             '_id': 'VgxgE3cBdDDj70WtOuFJ',
             '_score': 1.0,
             '_source': {'id': 74915741,
              'date': '2020-12-16T09:47:00',
              'actions': {'has_sessions': True,
               'has_basket': True,
               'order_screen': True,
               'purchased': False},
              'client': 'u_382139',
              'dimension': 'location_1',
              'promotion_id': None,
              'payment_amount': 52.75500000000001,
              'discount_amount': 0,
              'basket': {'p_10': {'price': 6.12, 'category': 'p_c_8', 'rank': 109},....},
              'total_products': 7,
              'session_start_date': '2020-12-16T09:39:11'}}

          Customers (Downloads) Index Document;
            {'id': 3840375,
             'download_date': '2020-12-31T14:56:32',
             'signup_date': '2021-01-02T13:55:32', 'client': 'u_344313'}
        """
        try:
            _insert = []
            data = data.to_dict('results')
            if index == 'orders':
                for i in data:
                    _obj = {i: None for i in orders_index_columns}
                    _keys = list(i.keys())
                    _has_purchased = True if i['has_purchased'] in ['True', True] else False
                    if len(self.actions[index]) != 0:
                        _obj['actions'] = {_a: False for _a in self.actions[index]}
                        _obj['actions']['purchased'] = _has_purchased
                        _obj['actions']['has_sessions'] = True
                    else:
                        _obj['actions'] = {'purchased': _has_purchased, 'has_sessions': True}
                    for k in _obj:
                        if k in _keys:
                            if k in ['date', 'session_start_date']:
                                _obj[k] = convert_to_iso_format(i[k])
                            else:
                                _obj[k] = i[k] if i[k] == i[k] and i[k] not in ['None', None, 'nan'] else None
                        else:
                            if k == 'id':
                                _obj['id'] = i['order_id']
                            if k == 'basket':
                                _obj['basket'] = i['basket'] if i['basket'] == i['basket'] else {}
                            if k == 'actions':
                                for a in self.actions[index]:
                                    if a in _keys:
                                        if i[a]:
                                            _obj['actions'][a] = True
                    _insert.append(_obj)
                    del _obj
                    if len(_insert) >= 10:
                        self.query_es.insert_data_to_index(_insert, index)
                        _insert = []

                if len(_insert) != 0:
                    self.query_es.insert_data_to_index(_insert, index)
                # insert logs into the sqlite logs table for sessions data insert process
                self.logs_update(logs={"page": "data-execute",
                                       "info": " SESSIONS index Done! - Number of documents :" + formating_numbers(len(data)),
                                       "color": "green"})
            _insert = []
            if index == 'downloads':
                for i in data:
                    _keys = list(i.keys())
                    _obj = {i: None for i in downloads_index_columns}
                    # TODO : id must be the counter not randomly selected
                    _obj['id'] = np.random.randint(200000000)
                    _obj['client'] = i['client']
                    _obj['download_date'] = convert_to_iso_format(i['download_date'])
                    if i.get('signup_date', None) is not None:
                        if i['signup_date'] not in ['nan', None, '', '-', 'Null']:
                            try:
                                _obj['signup_date'] = convert_to_iso_format(i['signup_date'])
                            except Exception as e_signup:
                                _obj['signup_date'] = None

                    for _a in self.actions[index]:
                        if i[_a] == i[_a]:
                            _obj[_a] = convert_to_iso_format(i[_a])

                    _insert.append(_obj)
                    if len(_insert) >= 10:
                        self.query_es.insert_data_to_index(_insert, index)
                        _insert = []

                if len(_insert) != 0:
                    self.query_es.insert_data_to_index(_insert, index)
                # insert logs into the sqlite logs table for customers data insert process
                self.logs_update(logs={"page": "data-execute",
                                       "info": " CUSTOMERS index Done! - Number of documents :" + formating_numbers(len(data)),
                                       "color": "green"})

        except Exception as e:
            try:
                err_str = " - " + str(e).replace("'", " ")
                err_str = err_str[0:100] if len(err_str) >= 100 else err_str
            except: err_str = " "
            self.logs_update(logs={"page": "data-execute",
                                   "info": "indexes Creation is failed! While " + index + " is creating. " + err_str,
                                   "color": "red"})

    def execute_index(self):
        """

        """
        self.job_start_date = current_date_to_day()
        self.get_schedule_data()
        self.get_end_date()

        try:
            for _data_type in ['orders', 'downloads']:
                if _data_type == 'orders':
                    _result_data = self.merge_orders(self.data_connection_structure['orders'],
                                                     self.data_connection_structure['products'])
                else:
                    _result_data = self.get_data(conf=self.data_connection_structure['downloads'],
                                                 data_source_type='downloads')
                if self.check_and_create_index(_data_type):
                    self.get_start_date()
                if len(_result_data) != 0:
                    self.insert_to_index(data=_result_data, index=_data_type)
                del _result_data

            last_schedule_triggered_date = str(current_date_to_day())
            self.update_schedule(date=last_schedule_triggered_date)

            self.end_date = current_date_to_day()

            spent_hour = round(abs(self.job_start_date - self.end_date).total_seconds() / 60 / 60, 2)
            total_time_str = str(round(spent_hour, 2)) + " hr. " if spent_hour >= 1 else str(round(spent_hour * 60, 2)) + " min. "
            comment = "indexes are created safely. Total spent time : " + total_time_str
            self.logs_update(logs={"page": "data-execute",
                                   "info": comment,
                                   "color": "green"})

        except Exception as e:
            try: err_str = " - " + str(e).replace("'", " ")
            except: err_str = " "
            self.logs_update(logs={"page": "data-execute",
                                   "info": "indexes Creation is failed!  " + err_str,
                                   "color": "red"})


