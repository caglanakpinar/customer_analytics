import sys, os, inspect, logging
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from sqlalchemy import create_engine, MetaData
from os.path import abspath, join

try: from utils import read_yaml, current_date_to_day, abspath_for_sample_data, sqlite_string_converter
except: from customeranalytics.utils import read_yaml, current_date_to_day, abspath_for_sample_data, sqlite_string_converter

try: from configs import query_path, default_es_port, default_es_host, default_message, schedule_columns
except: from customeranalytics.configs import query_path, default_es_port, default_es_host, default_message, schedule_columns


import pandas as pd
from datetime import datetime
from flask_login import current_user

try: from data_storage_configurations import connection_check, create_index, check_elasticsearch
except: from customeranalytics.data_storage_configurations import connection_check, create_index, check_elasticsearch

try: from exploratory_analysis import ea_configs
except: from customeranalytics.exploratory_analysis import ea_configs

try: from ml_process import ml_configs
except: from customeranalytics.ml_process import ml_configs


engine = create_engine('sqlite://///' + join(abspath_for_sample_data(), "web", 'db.sqlite3'), convert_unicode=True,
                       connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


class RouterRequest:
    def __init__(self):
        self.return_values = {}
        self.sqlite_queries = read_yaml(query_path, "queries.yaml")
        self.tables = pd.read_sql(self.sqlite_queries['tables'], con)
        self.table = None
        self.active_connections = False
        self.hold_connection = False
        self.recent_connection = False
        self.message = default_message
        self.success_data_execute = """ Data Storage Process is initialized!
                                        This process mainly involves fetching data
                                        from the data sources and storing them into the ElasticSearch indexes.
                                        This will take a while. Data Storage Process is triggered for
                                    """
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

    def delete_query(self, table, condition):
        _query = "DELETE FROM " + table
        _query += " WHERE " + condition
        _query = _query.replace("\\", "")
        return _query

    def check_for_table_exits(self, table):
        """
        checking sqlite if table is created before. If it not, table is created.
        :params table: checking table name in sqlite
        """
        if table not in list(self.tables['name']):
            con.execute(self.sqlite_queries[table])

    def logs_update(self, logs):
        """
        logs table in sqlite table is updated.
        chats table in sqlite table is updated.
        """
        try: self.check_for_table_exits(table='logs')
        except Exception as e: logging.error(e)

        try: self.check_for_table_exits(table='chat')
        except Exception as e: print(e)

        try:
            logs['login_user'] = current_user
            logs['log_time'] = str(current_date_to_day())[0:19]
            con.execute(self.insert_query(table='logs', columns=self.sqlite_queries['columns']['logs'][1:], values=logs))
        except Exception as e: logging.error(e)

        try: con.execute(self.insert_query(table='chat', columns=self.sqlite_queries['columns']['chat'][1:],
                                           values=self.info_logs_for_chat(logs['info'])))
        except Exception as e: print(e)

    def assign_color_for_es_tag(self, data):
        _unique_es_tags = list(data['tag'].unique())
        data = pd.merge(data, pd.DataFrame(zip(_unique_es_tags, self.colors[0:len(_unique_es_tags)])).rename(
            columns={0: "tag", 1: "color"}), on='tag', how='left')
        return data

    def collect_data_from_table(self, table, query_str=None):
        data = pd.DataFrame()
        try:
            data = pd.read_sql("SELECT * FROM " + table, con)
            if query_str is not None:
                data = data.query(query_str)
        except Exception as e: logging.error(e)
        return data

    def get_intersect_columns_with_request(self, requests, table):
        return list(set(list(requests.keys())) & set(self.sqlite_queries['columns'][table][1:]))

    def check_for_data_source_connection(self, requests, columns):
        try:
            return connection_check(request={col: requests[col] for col in columns},
                                    index=requests['data_type'],
                                    type=requests['data_type'])
        except Exception as e: logging.error(e)

    def check_for_both_sessions_and_customers_data_source(self, data_connection):
        if data_connection['orders_data_source_tag'] not in ['None', None] and \
                data_connection['downloads_data_source_tag'] not in ['None', None]:
            return True
        else: return False

    def check_for_product_data_source(self, data_connection):
        if data_connection['products_data_source_tag'] not in ['None', None]:
            return True
        else: return False

    def check_for_insert_columns(self, columns, requests, table):
        try:
            for col in self.sqlite_queries['columns'][table][1:]:
                if col not in columns:
                    requests[col] = None
        except Exception as e: logging.error(e)
        return requests

    def check_for_session_and_customer_product_connect(self):
        data_connection = self.collect_data_from_table(table='data_connection')
        sessions, customers, products = False, False, False
        if len(data_connection) != 0:
            data_connection = data_connection.to_dict('results')[-1]
            sessions = True if data_connection['orders_data_source_tag'] != 'None' else False
            customers = True if data_connection['downloads_data_source_tag'] != 'None' else False
            products = True if data_connection['products_data_source_tag'] != 'None' else False
        return sessions, customers, products

    def values_for_manage_data(self, template):
        es_connection = self.collect_data_from_table(table='es_connection')
        if len(es_connection) != 0:
            if template in ['add-data-purchase_2', 'add-data-product_2']:
                self.message['es_connection'] = es_connection.to_dict('results')[-1]
            else:
                self.message['es_connection'] = es_connection.to_dict('results')
            self.message['s_c_p_connection_check'] = "_".join(
                [str(i) for i in self.check_for_session_and_customer_product_connect()])

    def values_for_schedule_data(self):
        try:
            data_connection = self.collect_data_from_table(table='data_connection')
        except Exception as e:
            logging.error(e)
        prev_schedule = self.collect_data_from_table(table='schedule_data')
        if len(prev_schedule) != 0:
            self.message['schedule_check'] = True
        actions = self.collect_data_from_table(table='actions')
        if len(actions) != 0:
            actions = actions.groupby("data_type").agg(
                {"action_name": lambda x: ", ".join(list(x))}).reset_index().fillna('....')
        try:
            es_connection = self.collect_data_from_table(table='es_connection')
            if len(es_connection) != 0:
                self.message['es_connection'] = es_connection.to_dict('results')[-1]
            else:
                self.message['es_connection'] = '....'
        except Exception as e: logging.error(e)

        try:
            logs = self.collect_data_from_table(table='logs')
            if len(logs) != 0:
                logs['color'] = logs['color'].apply(lambda x: 'color:' + x + ';')
                self.message['logs'] = logs.to_dict('results')[-min(len(logs), 20):]
            else:
                self.message['logs'] = '....'
        except Exception as e: logging.error(e)

        if len(data_connection) != 0:
            if self.check_for_both_sessions_and_customers_data_source(data_connection.to_dict('results')[-1]):
                self.message['connect_accept'] = True
                for dt in ['orders', 'downloads', 'products']:
                    data_connection[dt + '_data_query_path'] = sqlite_string_converter(
                        list(data_connection[dt + '_data_query_path'])[0], back_to_normal=True)
                data_connection = pd.concat([data_connection, prev_schedule], axis=1)
                data_connection = pd.concat([data_connection,
                                             actions.query("data_type == 'orders'").drop('data_type', axis=1).rename(
                                                 columns={"action_name": "ses_actions"})], axis=1).fillna('....')
                data_connection = pd.concat([data_connection,
                                             actions.query("data_type == 'downloads'").drop('data_type', axis=1).rename(
                                                 columns={"action_name": "d_actions"})], axis=1).fillna('....')

                schedule = data_connection.to_dict('results')[-1]
                self.message['schedule'] = {i: '....' for i in list(schedule.keys()) + ['ses_actions', 'd_actions'] +
                                            self.sqlite_queries['columns']['schedule_data'][1:]}
                for i in schedule:
                    if schedule[i] not in [None, 'None']:
                        self.message['schedule'][i] = schedule[i]
                self.message['schedule'] = [self.message['schedule']]
                if self.check_for_product_data_source(data_connection.to_dict('results')[-1]):
                    self.message['has_product_data_source'] = True

    def update_data_query_path_for_insert(self, requests):
        _data_type = requests['data_type']
        requests[_data_type + '_data_query_path'] = sqlite_string_converter(requests[_data_type + '_data_query_path'])
        return requests

    def update_data_connection_table(self, requests, columns):
        data_connections = self.collect_data_from_table(table='data_connection')
        requests = self.update_data_query_path_for_insert(requests)
        if len(data_connections) == 0:
            for col in self.sqlite_queries['columns']['data_connection'][1:]:
                if col not in list(requests.keys()):
                    requests[col] = None
            try:
                con.execute(self.insert_query(table='data_connection',
                                              columns=self.sqlite_queries['columns']['data_connection'][1:],
                                              values=requests))
            except Exception as e: logging.error(e)
        else:
            data_connections = data_connections.to_dict('results')[-1]
            try:
                con.execute(self.update_query(table='data_connection',
                                              condition=" id = " + str(data_connections['id']),
                                              columns=columns, values=requests))
            except Exception as e: logging.error(e)

    def update_data_columns_match_table(self, requests, columns):
        try:
            self.check_for_table_exits(table='data_columns_integration')
            data_columns_integration = self.collect_data_from_table(table='data_columns_integration', query_str=" id == 1")
            if len(data_columns_integration) == 0:
                requests = self.check_for_insert_columns(columns, requests, 'data_columns_integration')
                try:
                    con.execute(self.insert_query(table='data_columns_integration',
                                                  columns=self.sqlite_queries['columns']['data_columns_integration'][1:],
                                                  values=requests))
                except Exception as e: logging.error(e)
            else:
                try:
                    con.execute(self.update_query(table='data_columns_integration',
                                                  condition=" id = 1 ",
                                                  columns=columns, values=requests))
                except Exception as e: logging.error(e)
        except Exception as e: logging.error(e)

    def remove_data_type_action(self, requests):
        prev_actions = self.collect_data_from_table(table='actions')
        if len(prev_actions) != 0:
            prev_actions_data_type = prev_actions[prev_actions['data_type'] == requests['data_type']]
            if len(prev_actions_data_type) != 0:
                for a in prev_actions_data_type.to_dict('results'):
                    con.execute(self.delete_query(table='actions',
                                                  condition=" id = " + str(a['id'])))

    def update_actions_table(self, requests):
        if requests['data_type'] != 'products':
            if requests.get('actions', None) is not None:
                self.check_for_table_exits(table='actions')
                self.remove_data_type_action(requests)
                actions = []
                if requests['actions'] != '':
                    for i in [i.replace(" ", "") for i in requests['actions'].split(",")]:
                        counter = 0
                        for c in i:
                            if c == ' ':
                                counter += 1
                            else:
                                break
                        _action = i[counter:]
                        actions.append(_action)
                        con.execute(self.insert_query(table='actions',
                                                      columns=self.sqlite_queries['columns']['actions'][1:],
                                                      values={"action_name": _action, "data_type": requests['data_type']}))
                    requests['actions'] = ",".join(actions)
        else:
            self.remove_data_type_action(requests)

    def update_schedule_table(self, requests):
        try:
            self.check_for_table_exits(table='schedule_data')
            prev_schedule = self.collect_data_from_table(table='schedule_data').to_dict('results')
            if len(prev_schedule) != 0:
                self.logs_update(logs={"page": "data-execute", "info": "Previous job " + " is removed.", "color": "red"})
                con.execute(self.delete_query(table='schedule_data', condition=" id = 1"))
            columns = self.get_intersect_columns_with_request(requests, 'schedule_data')
            requests = self.check_for_insert_columns(columns, requests, 'schedule_data')
            requests['max_date_of_order_data'] = str(current_date_to_day())[0:19]
            con.execute(self.insert_query(table='schedule_data',
                                          columns=self.sqlite_queries['columns']['schedule_data'][1:],
                                          values=requests))
            self.logs_update(logs={"page": "data-execute",
                                   "info": self.success_data_execute + " ".join(requests['time_period'].split("_")),
                                   "color": "green"})
        except Exception as e:
            logging.error(e)
        return requests['tag']

    def update_data_query_path_on_schedule(self, request):
        keys = list(request.keys())
        data_source_query_path = [i for i in ['orders', 'downloads', 'products'] if i + '_data_query_path' in keys]
        con.execute(self.update_query(table='data_columns_integration',
                                      condition=" id = 1 ",
                                      columns=data_source_query_path,
                                      values={data_source_query_path: request[data_source_query_path[0]]}))

    def update_message_and_tables(self):
        self.tables = pd.read_sql(self.sqlite_queries['tables'], con)
        self.message = default_message

    def manage_data_integration(self, requests):
        if requests.get('connect', None) is not None:
            self.check_for_table_exits(table='es_connection')
            requests['port'] = str(default_es_port) if requests['port'] is None else requests['port']
            requests['host'] = str(default_es_host) if requests['host'] is None else requests['host']
            status, self.message['es_connection_check'] = check_elasticsearch(port=requests['port'],
                                                                              host=requests['host'],
                                                                              directory=requests['directory'])
            if status:
                try:
                    con.execute(self.insert_query(table='es_connection',
                                                  columns=self.sqlite_queries['columns']['es_connection'][1:],
                                                  values=requests))
                except Exception as e:
                    logging.error(e)

        if requests.get('delete', None) is not None:
            try:
                con.execute("DROP table es_connection")
            except Exception as e:
                logging.error(e)

    def data_connections(self, requests):
        # for orders index choose ElasticSearch Connection from es_connection table (only status == 'on')
        if requests.get('connect', None) is not None:
            self.check_for_table_exits(table='data_connection')
            self.check_for_table_exits(table='data_columns_integration')
            _columns = self.get_intersect_columns_with_request(requests, 'data_connection')
            _columns_2 = self.get_intersect_columns_with_request(requests, 'data_columns_integration')
            conn_status, self.message['data_source_con_check'], data, data_columns = self.check_for_data_source_connection(requests, _columns)

            # connection update
            if conn_status:
                self.update_actions_table(requests)
                self.update_data_connection_table(requests, _columns)
                self.update_data_columns_match_table(requests, _columns_2)

    def data_execute(self, requests):
        if requests.get('schedule', None) is not None:
            es_tag = self.update_schedule_table(requests)
            create_index(tag=es_tag, ea_configs=ea_configs, ml_configs=ml_configs)
        if requests.get('edit', None) is not None:
            self.update_data_query_path_on_schedule(requests)
        if requests.get('delete', None) is not None:
            try:
                con.execute("DELETE FROM schedule_data")
            except Exception as e:
                logging.error(e)

    def check_for_request(self, _r):
        _r_updated = {}
        try:
            for i in _r:
                _r_updated[i] = _r[i]
                if _r_updated[i] == 'True':
                    _r_updated[i] = True
                if _r_updated[i] == 'None':
                    _r_updated[i] = None
                if i == 'connect':
                    if _r_updated[i] not in [True, 'True']:
                        _r_updated['data_type'] = _r[i]
                        _r_updated['connect'] = True

                if i == 'schedule':
                    _r_updated['es_tag'] = _r[i]
                    _r_updated['schedule'] = True

        except Exception as e:
            logging.error(e)
        return _r_updated

    def execute_request(self, req, template):
        if req != {}:
            if template == 'data-es':
                self.manage_data_integration(self.check_for_request(req))
            if template in ['add-data-purchase', 'add-data-product']:
                self.data_connections(self.check_for_request(req))
            if template == 'data-execute':
                self.data_execute(self.check_for_request(req))

    def fetch_results(self, template, requests):
        if requests == {}:
            self.update_message_and_tables()
        else:
            if 'delete' in list(requests.keys()):
                self.update_message_and_tables()

        if template in ['add-data-purchase', 'add-data-product']:
            self.values_for_manage_data(template)
        if template == 'data-es':
            try:
                es_connection = self.collect_data_from_table(table='es_connection')
                if len(es_connection) != 0:
                    self.message['es_connection'] = es_connection.to_dict('results')[-1]
                else:
                    self.message['es_connection'] = '....'
            except Exception as e:
                logging.error(e)
        if template == 'data-execute':
            self.values_for_schedule_data()



