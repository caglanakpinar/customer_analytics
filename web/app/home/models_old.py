import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)
#
# from sqlalchemy import create_engine, MetaData
# from os.path import abspath, join
# from utils import read_yaml, rgb_to_hex, current_date_to_day
# from configs import query_path, default_es_port, default_es_host, default_message, schedule_columns
# import pandas as pd
# from numpy import array, random
#
# from data_storage_configurations import connection_check, create_index, check_elasticsearch
#
#
# engine = create_engine('sqlite://///' + join(abspath(""), "web", 'db.sqlite3'), convert_unicode=True, connect_args={'check_same_thread': False})
# metadata = MetaData(bind=engine)
# con = engine.connect()
#
#
# class RouterRequest:
#     def __init__(self):
#         self.return_values = {}
#         self.sqlite_queries = read_yaml(query_path, "queries.yaml")
#         self.tables = pd.read_sql(self.sqlite_queries['tables'], con)
#         self.table = None
#         self.active_connections = False
#         self.hold_connection = False
#         self.recent_connection = False
#         self.message = default_message
#         self.colors = ["color:" + rgb_to_hex(tuple(random.choice(range(256) , size=3))) + ";" for i in range(100)]
#
#     def insert_query(self, table, columns, values):
#         values = [values[col] for col in columns]
#         _query = "INSERT INTO " + table + " "
#         _query += " (" + ", ".join(columns) + ") "
#         _query += " VALUES (" + ", ".join([" '{}' ".format(v) for v in values]) + ") "
#         _query = _query.replace("\\", "")
#         return _query
#
#     def update_query(self, table, condition, columns, values):
#         values = [(col, values[col]) for col in columns if values.get(col, None) is not None]
#         _query = "UPDATE " + table
#         _query += " SET " + ", ".join([i[0] + " = '" + i[1] + "'" for i in values])
#         _query +=" WHERE " + condition
#         _query = _query.replace("\\", "")
#         print(_query)
#         return _query
#
#     def delete_query(self, table, condition):
#         _query = "DELETE FROM " + table
#         _query += " WHERE " + condition
#         _query = _query.replace("\\", "")
#         return _query
#
#     def check_for_table_exits(self, table, query=None):
#         if table not in list(self.tables['name']):
#             if query is None:
#                 con.execute(self.sqlite_queries[table])
#             else:
#                 con.execute(query)
#
#     def check_for_sample_tables(self):
#         accept = False
#         for i in ['', 'action_', 'product_', 'promotion_']:
#             for ind in ['orders_sample_data', 'downloads_sample_data']:
#                 if i + 'orders_sample_data' in list(self.tables['name']):
#                     accept = True
#         return accept
#
#     def logs_update(self, logs):
#         self.check_for_table_exits(table='logs')
#         con.execute(self.insert_query(table='logs',
#                                       columns=self.sqlite_queries['columns']['logs'][1:],
#                                       values=logs
#                                       ))
#
#     def query_data_connection(self, requests):
#         data_type, type_of_data_type, conn_values = None, '', {}
#         if requests.get('orders_data_query_path', None) is not None:
#             data_type = 'orders'
#         if requests.get('downloads_data_query_path', None) is not None:
#             data_type = 'downloads'
#         if data_type is not None:
#             conn_values = pd.read_sql(""" SELECT
#                                              id,
#                                              %(data_type)s_data_source_tag,
#                                              %(data_type)s_data_source_type,
#                                              %(data_type)s_data_query_path,
#                                              %(data_type)s_password,
#                                              %(data_type)s_user,
#                                              %(data_type)s_port,
#                                              %(data_type)s_host,
#                                              is_action,
#                                              is_product,
#                                              is_promotion
#                                           FROM data_connection
#                                           WHERE %(data_type)s_data_source_tag = '""" % {'data_type': data_type} +
#                                       str(requests['resent_tag']) + "' ", con).to_dict('results')[0]
#             conn_values[data_type + '_data_query_path'] = requests[data_type + '_data_query_path']
#
#         type_of_data_type = data_type
#         for types in ['is_action', 'is_product', 'is_promotion']:
#             if conn_values[types]:
#                 type_of_data_type = types
#         return conn_values, data_type, type_of_data_type
#
#     def columns_matching_decision(self, row):
#         decision = 'not assigned'
#         if row['is_action'] == 'True':
#             if row['orders_id'] != '....' or row['downloads_id'] != '....':
#                 decision = 'assigned'
#         if row['is_product'] == 'True' or row['is_promotion'] == 'True':
#             if row['orders_id'] == row['orders_id'] and row['orders_id'] not in ['....', 'None', None]:
#                 decision = 'assigned'
#         if row['is_product'] != 'True' and row['is_promotion'] != 'True' and row['is_action'] != 'True':
#             total_assigned = sum([1 for i in [row['orders_id'], row['downloads_id']] if i not in ['....', 'None', None]])
#             if total_assigned == 2:
#                 decision = 'assigned'
#             if total_assigned == 0:
#                 decision = 'not assigned'
#             if total_assigned == 1:
#                 if row['orders_id'] == row['orders_id']:
#                     decision = 'not assigned - customers'
#                 else: decision = 'not assigned - sessions'
#         return decision
#
#     def assign_color_for_es_tag(self, data):
#         _unique_es_tags = list(data['tag'].unique())
#         data = pd.merge(data, pd.DataFrame(zip(_unique_es_tags, self.colors[0:len(_unique_es_tags)])).rename(
#             columns={0: "tag", 1: "color"}), on='tag', how='left')
#         return data
#
#     def data_connections_hold_edit_connection_check(self):
#         conns = pd.read_sql(
#                             """ SELECT
#                                     id, tag, process
#                                 FROM data_connection
#                                 WHERE process in ('hold', 'edit', 'add_dimension',
#                                                   'add_action', 'add_product', 'add_promotion') """, con)
#         if len(conns) != 0:
#             holds = conns.query("process == ('hold', 'add_dimension', 'add_action', 'add_product', 'add_promotion')")
#             edits = conns.query("process == 'edit'")
#             if len(holds) != 0:
#                 for _id in list(holds['id'].unique()):
#                     con.execute(self.delete_query(table='data_connection', condition=" id = " + str(_id) + " "))
#             if len(edits) != 0:
#                 for _id in list(edits['id'].unique()):
#                     con.execute(self.update_query(table='data_connection',
#                                                   columns=["process"],
#                                                   values={"process": "connected"},
#                                                   condition=" id = '" + str(_id) + "' "))
#
#     def sample_data_insert(self,
#                            is_for_orders,
#                            is_for_action=False,
#                            is_for_product=False,
#                            is_for_promotion=False,
#                            data=None):
#         if data is not None:
#             insert_columns = list(data[0].keys())
#             if is_for_orders:
#                 table = 'orders_sample_data'
#                 self.message['orders_data'] = data
#             else:
#                 table = 'downloads_sample_data'
#                 self.message['downloads_data'] = data
#
#             if is_for_action:
#                 table = 'action_' + table
#
#             if is_for_product:
#                 table = 'product_' + table
#
#             if is_for_promotion:
#                 table = 'promotion_' + table
#
#             try:
#                 con.execute("DROP TABLE " + table)
#             except Exception as e:
#                 print(e)
#             con.execute(" CREATE TABLE " + table + " (" + " VARCHAR, ".join(insert_columns) + ")")
#             for i in data:
#                 try:
#                     con.execute(self.insert_query(table=table,
#                                                   columns=insert_columns,
#                                                   values=i))
#                 except Exception as e:
#                     print(e)
#
#     def sample_data_column_insert(self, is_for_orders, tag, columns, type=''):
#         self.check_for_table_exits(table='data_columns')
#         data_type = 'orders' if is_for_orders else 'downloads'
#         tag = tag[data_type + '_data_source_tag']
#         try:
#             con.execute(self.insert_query(table='data_columns',
#                                           columns=self.sqlite_queries['columns']['data_columns'][1:],
#                                           values={'tag': tag,
#                                                   'data_type': type + data_type,
#                                                   'columns': "*".join(columns.tolist())}))
#
#         except Exception as e:
#             print(e)
#
#     def get_holded_connection(self, source_tag_name):
#         tags = pd.read_sql(
#             """
#             SELECT
#             id, tag, process, dimension, """ + source_tag_name +
#             """
#             FROM data_connection
#             WHERE process in ('hold', 'edit', 'add_dimension', 'add_action',
#                               'add_product', 'add_promotion') AND dimension != 'sample_data'
#             """, con).tail(1)
#         id, process,  = list(tags['id'])[0], list(tags['process'])[0]
#         source_tag_name, dimension = list(tags[source_tag_name])[0], list(tags['dimension'])[0]
#         return id, process, source_tag_name, dimension
#
#     def check_for_request(self, _r):
#         _r_updated = {}
#         try:
#             for i in _r:
#                 if _r[i] != _r[i] or _r[i] is None or _r[i] in ['', '-', '....']:
#                     if 'edit' not in list(_r.keys()):
#                         _r_updated[i] = None
#                 else:
#                     if i in ['delete', 'edit', 'connect']:
#                         if _r[i] != 'True':
#                             if i == 'edit':
#                                 _r_updated['recent_tag'] = _r[i]
#                             else:
#                                 _r_updated['tag'] = _r[i]
#                         _r_updated[i] = 'True'
#                     else:
#                         if i == 'process':
#                             _r_updated[_r[i]] = 'True'
#                         else:
#                             _r_updated[i] = _r[i]
#
#         except Exception as e:
#             print(e)
#         return _r_updated
#
#     def manage_data_integration(self, requests):
#         if requests.get('connect', None) == 'True':
#             self.check_for_table_exits(table='es_connection')
#             requests['port'] = str(default_es_port) if requests['port'] is None else requests['port']
#             requests['host'] = str(default_es_host) if requests['host'] is None else requests['host']
#             requests['status'] = 'on'
#             status, self.message['es_connection'] = check_elasticsearch(port=requests['port'],
#                                                                         host=requests['host'],
#                                                                         directory=requests['directory'])
#
#             if status:
#                 try:
#                     con.execute(self.insert_query(table='es_connection',
#                                                   columns=self.sqlite_queries['columns']['es_connection'][1:],
#                                                   values=requests))
#                 except Exception as e:
#                     print(e)
#
#         if requests.get('edit', None) == 'True':
#             conn = pd.read_sql("SELECT * es_connection FROM WHERE tag = '" + requests['recent_tag'] + "'", con)
#             args = {a: list(conn[a])[0] for a in ['port', 'host', 'directory']}
#             for a in ['port', 'host', 'directory']:
#                 if requests.get(a, None) not in [None, 'None', '....']:
#                     args[a] = requests[a]
#             status, self.message['es_connection'] = check_elasticsearch(**args)
#             if status:
#                 try:
#                     con.execute(self.update_query(table='es_connection',
#                                                   condition=" tag = '" + requests['recent_tag'] + "' ",
#                                                   columns=["status", "port", "host", "tag"],
#                                                   values=requests))
#                 except Exception as e:
#                     print(e)
#
#         if requests.get('delete', None) == 'True':
#             try:
#                 print()
#                 con.execute(self.delete_query(table='es_connection',
#                                               condition=" tag = '" + requests['tag'] + "' "))
#             except Exception as e:
#                 print(e)
#         self.tables = pd.read_sql(self.sqlite_queries['tables'], con)
#
#     def data_connections(self, requests):
#         self.message = default_message
#         # for orders index choose ElasticSearch Connection from es_connection table (only status == 'on')
#         if requests.get('connect', None) == 'True':
#             self.check_for_table_exits(table='data_connection')
#             for col in self.sqlite_queries['columns']['data_connection'][1:]:
#                 if col not in list(requests.keys()):
#                     requests[col] = None
#             requests['process'] = 'hold'
#             self.data_connections_hold_edit_connection_check()
#             try:
#                 con.execute(self.insert_query(table='data_connection',
#                                               columns=self.sqlite_queries['columns']['data_connection'][1:],
#                                               values=requests))
#                 print()
#             except Exception as e:
#                 print(e)
#
#         if requests.get('cancel', None) == 'True':
#             self.data_connections_hold_edit_connection_check()
#
#         if requests.get('edit', None) == 'True':
#             requests['process'] = 'edit'
#             self.data_connections_hold_edit_connection_check()
#             con.execute(self.update_query(table='data_connection',
#                                           condition=" id = " + str(requests['id']),
#                                           columns=["process"],
#                                           values=requests))
#
#         if requests.get('add_dimension', None) == 'True' or \
#            requests.get('add_action', None) == 'True' or \
#            requests.get('add_product', None) == 'True' or \
#            requests.get('add_promotion', None) == 'True':
#             for col in self.sqlite_queries['columns']['data_connection'][1:]:
#                 if col not in list(requests.keys()):
#                     requests[col] = None
#             if requests.get('add_dimension', None) == 'True':
#                 requests['process'] = 'add_dimension'
#                 requests['dimension'] = 'True'
#             if requests.get('add_action', None) == 'True':
#                 requests['is_action'] = 'True'
#                 requests['process'] = 'add_action'
#             if requests.get('add_product', None) == 'True':
#                 requests['is_product'] = 'True'
#                 requests['process'] = 'add_product'
#             if requests.get('add_promotion', None) == 'True':
#                 requests['is_promotion'] = 'True'
#                 requests['process'] = 'add_promotion'
#             recent_connection = pd.read_sql("""
#                                             SELECT tag, dimension
#                                             FROM data_connection WHERE id =
#                                             """ + str(requests['id']),
#                                                con).to_dict('results')[0]
#
#             insert_available = True
#             requests['tag'] = recent_connection['tag']
#             if 'True' in [requests.get('add_action', None),
#                           requests.get('is_product', None),
#                           requests.get('is_promotion', None)]:
#                 if recent_connection['dimension'] != 'None':
#                     requests['dimension'] = requests['id']
#             else:
#                 if recent_connection['dimension'] != 'None':
#                     insert_available = False
#
#             if insert_available:
#                 self.data_connections_hold_edit_connection_check()
#                 con.execute(self.insert_query(table='data_connection',
#                                               columns=self.sqlite_queries['columns']['data_connection'][1:],
#                                               values=requests))
#
#         if requests.get('delete', None) == 'True':
#             con.execute(self.delete_query(table='data_connection', condition=" id = " + str(requests['id'])))
#
#         if requests.get('orders_edit', None) == 'True' or \
#            requests.get('downloads_edit', None) == 'True' or \
#            requests.get('actions_orders_edit', None) == 'True' or \
#            requests.get('actions_downloads_edit', None) == 'True' or \
#            requests.get('products_orders_edit', None) == 'True' or \
#            requests.get('promotions_orders_edit', None) == 'True':
#             source_tag_name = 'orders_data_source_tag'
#             is_for_orders, is_for_action, is_for_product, is_for_promotion = False, False, False, False
#             type_of_data_type = ''
#             if requests.get('orders_edit', None) == 'True' or \
#                requests.get('actions_orders_edit', None) == 'True' or \
#                requests.get('products_orders_edit', None) == 'True':
#                 source_tag_name = 'downloads_data_source_tag'
#                 is_for_orders = True
#             tags = pd.read_sql(
#                 """
#                 SELECT
#                 id, tag, process, dimension, """ + source_tag_name +
#                 """
#                 FROM data_connection
#                 WHERE process in ('hold', 'edit', 'add_dimension', 'add_action', 'add_product', 'add_promotion')
#                                           AND dimension != 'sample_data'
#                 """, con).tail(1)
#             id, process, tag = list(tags['id'])[0], list(tags['process'])[0], list(tags[source_tag_name])[0]
#             requests['process'] = process
#             if requests.get('actions_orders_edit', None) == 'True' or requests.get('actions_downloads_edit', None) == 'True':
#                 is_for_orders = True if requests.get('actions_orders_edit', None) == 'True' else False
#                 is_for_action = True
#                 type_of_data_type = 'action_'
#                 requests['is_action'] = 'True'
#                 requests['process'] = 'connected'
#             if requests.get('products_orders_edit', None) == 'True':
#                 is_for_product, is_for_orders = True, True
#                 type_of_data_type = 'product_'
#                 requests['is_product'] = 'True'
#                 requests['process'] = 'connected'
#             if requests.get('promotions_orders_edit', None) == 'True':
#                 is_for_promotion, is_for_orders = True, True
#                 type_of_data_type = 'promotion_'
#                 requests['is_promotion'] = 'True'
#                 requests['process'] = 'connected'
#             if requests.get('orders_edit', None) == 'True' or requests.get('downloads_edit', None) == 'True':
#                 if list(tags[source_tag_name])[0] not in ['', 'None', None, 'Null']:
#                     requests['process'] = 'connected'
#
#             _columns = list(set(list(requests.keys())) & set(self.sqlite_queries['columns']['data_connection'][1:]))
#             conn_status, message, data, data_columns = connection_check(request={col: requests[col] for col in _columns},
#                                                                         index='orders' if is_for_orders else 'downloads',
#                                                                         type=type_of_data_type)
#
#             self.message['orders'] = message if is_for_orders else '....'
#             self.message['downloads'] = message if not is_for_orders else '....'
#             self.message['orders_columns'] = data_columns if is_for_orders else '....'
#             self.message['downloads_columns'] = data_columns if not is_for_orders else '....'
#             if is_for_action:
#                 self.message['action_orders'] = message if is_for_orders else '....'
#                 self.message['action_downloads'] = message if not is_for_orders else '....'
#             if is_for_product:
#                 self.message['product_orders'] = message
#             if is_for_promotion:
#                 self.message['promotion_orders'] = message
#
#             if conn_status:
#                 try:
#                     con.execute(self.update_query(table='data_connection',
#                                                   condition=" id = " + str(id) + " ",
#                                                   columns=_columns,
#                                                   values=requests))
#                 except Exception as e:
#                     print(e)
#
#                 self.sample_data_insert(is_for_orders=is_for_orders,
#                                         is_for_action=is_for_action,
#                                         is_for_product=is_for_product,
#                                         is_for_promotion=is_for_promotion,
#                                         data=data)
#                 self.sample_data_column_insert(is_for_orders=is_for_orders,
#                                                tag=requests,
#                                                columns=data_columns,
#                                                type=type_of_data_type)
#
#         if requests.get('orders_column_replacement', None) == 'True' or \
#            requests.get('downloads_column_replacement', None) == 'True' or \
#            requests.get('action_orders_column_replacement', None) == 'True' or \
#            requests.get('action_downloads_column_replacement', None) == 'True' or \
#            requests.get('product_orders_column_replacement', None) == 'True' or \
#            requests.get('promotion_orders_column_replacement', None) == 'True':
#             source_tag_name, data_type, s_table = 'downloads_data_source_tag', 'downloads', 'downloads_sample_data'
#             is_for_orders, is_for_action, is_for_product, is_for_promotion = False, False, False, False
#             if requests.get('orders_column_replacement', None) == 'True':
#                 source_tag_name, data_type, s_table = 'orders_data_source_tag', 'orders', 'orders_sample_data'
#                 is_for_orders = True
#             if requests.get('action_orders_column_replacement', None) == 'True' or \
#                requests.get('action_downloads_column_replacement', None) == 'True':
#                 source_tag_name, data_type = 'downloads_data_source_tag', 'action_downloads'
#                 s_table, is_for_action = 'action_downloads_sample_data', True
#                 if requests.get('action_orders_column_replacement', None) == 'True':
#                     is_for_orders = True
#                     source_tag_name, data_type,  = 'orders_data_source_tag', 'action_orders'
#                     s_table, is_for_action = 'action_orders_sample_data', True
#             if requests.get('product_orders_column_replacement', None) == 'True':
#                 source_tag_name, data_type, = 'orders_data_source_tag', 'product_orders'
#                 s_table, is_for_product = 'product_orders_sample_data', True
#                 is_for_orders = True
#             if requests.get('promotion_orders_column_replacement', None) == 'True':
#                 source_tag_name, data_type, = 'orders_data_source_tag', 'promotion_orders'
#                 s_table, is_for_promotion = 'promotion_orders_sample_data', True
#                 is_for_orders = True
#             id, process, requests['tag'], dimension = self.get_holded_connection(source_tag_name)
#
#             if dimension != 'None' and (requests.get('action_orders_column_replacement', None) == 'True' or \
#                                         requests.get('action_downloads_column_replacement', None) == 'True'):
#                 data_type = "*".join([data_type, str(dimension)])
#
#             self.check_for_table_exits(table='data_columns_integration')
#             data_columns_integration = pd.read_sql(""" SELECT id
#                                                        FROM data_columns_integration
#                                                        WHERE tag = '""" + requests['tag'] +
#                                                    "'  AND data_type = '" + data_type + "' ",
#                                                    con).to_dict('results')
#             requests['data_type'] = data_type
#             if len(data_columns_integration) == 0:  # insert into data_columns_integration
#                 for col in self.sqlite_queries['columns']['data_columns_integration'][1:]:
#                     if col not in list(requests.keys()):
#                         requests[col] = None
#                 con.execute(self.insert_query(table='data_columns_integration',
#                                               columns=self.sqlite_queries['columns']['data_columns_integration'][1:],
#                                               values=requests))
#             else:
#                 con.execute(self.update_query(table='data_columns_integration',
#                                               condition=" id = " + str(data_columns_integration[0]['id']) + " ",
#                                               columns=self.sqlite_queries['columns']['data_columns_integration'][1:],
#                                               values=requests))
#
#             try:
#                 _sample_data_table = pd.read_sql(""" SELECT * FROM  """ + s_table, con)
#                 _sample_data_table = _sample_data_table.rename(columns={requests[i]: i for i in requests})
#                 self.sample_data_insert(is_for_orders=is_for_orders,
#                                         is_for_action=is_for_action,
#                                         is_for_product=is_for_product,
#                                         is_for_promotion=is_for_promotion,
#                                         data=_sample_data_table.to_dict('results'))
#             except Exception as e:
#                 print(e)
#
#             self.message[data_type] = 'Connected!'
#
#     def create_sample_data(self, requests):
#         # for orders index choose ElasticSearch Connection from es_connection table (only status == 'on')
#         if requests.get('connect', None) == 'True':
#             self.check_for_table_exits(table='data_connection')
#             for col in self.sqlite_queries['columns']['data_connection'][1:]:
#                 if col not in list(requests.keys()):
#                     requests[col] = None
#             requests['process'] = 'connected'
#             requests['dimension'] = 'sample_data'
#             try:
#                 con.execute(self.insert_query(table='data_connection',
#                                               columns=self.sqlite_queries['columns']['data_connection'][1:],
#                                               values=requests))
#                 create_sample_data(requests['tag'])
#             except Exception as e:
#                 print(e)
#
#     def data_execute(self, requests):
#         logs = {'page': 'data-execute', 'color': 'red', 'info': 'Failed!'}
#         if requests.get('schedule', None) == 'True':
#             self.check_for_table_exits(table='schedule_data')
#             prev_schedule = pd.DataFrame()
#             try:
#                 prev_schedule = pd.read_sql(""" SELECT * FROM schedule_data  WHERE tag = '""" + requests['tag'] + "' ",
#                                             con).to_dict('results')
#             except Exception as e:
#                 print(e)
#
#             if len(prev_schedule) == 0:
#                 for r in self.sqlite_queries['columns']['schedule_data'][1:]:
#                     if r not in list(requests.keys()):
#                         requests[r] = None
#                 requests['status'] = 'on'
#                 requests['max_date_of_order_data'] = str(current_date_to_day())[0:19]
#                 try:
#                     con.execute(self.insert_query(table='schedule_data',
#                                                   columns=self.sqlite_queries['columns']['schedule_data'][1:],
#                                                   values=requests))
#                     create_index(tag=requests['tag'])
#                 except Exception as e:
#                     print(e)
#             else:
#                 con.execute(self.update_query(table='schedule_data',
#                                               condition=" tag = '" + requests['tag'] + "' ",
#                                               columns=["status"],
#                                               values={'status': 'on'}))
#                 create_index(tag=requests['tag'])
#
#         if requests.get('delete', None) == 'True':
#             con.execute(self.delete_query(table='schedule_data', condition=" tag = '" + requests['tag'] + "' "))
#
#         if requests.get('edit', None) == 'True':
#             logs = {'page': 'data-execute', 'color': 'red', 'info': 'Failed!'}
#             conn, data_type, type_of_data_type = self.query_data_connection(requests=requests)
#             if len(conn) != 0:
#                 conn_status, message, data, data_columns = connection_check(
#                     request=conn,
#                     index=data_type,
#                     type=type_of_data_type)
#                 if conn_status:
#                     try:
#                         con.execute(self.update_query(table='data_connection',
#                                                       condition=" id = " + str(conn['id']) + " ",
#                                                       columns=[data_type + '_data_query_path'],
#                                                       values={data_type + '_data_query_path':
#                                                                   requests[data_type + '_data_query_path']}))
#                         logs['color'], logs['info'] = 'green', "Connected Successfully!! Data Query/Path is updated."
#                     except Exception as e:
#                         print(e)
#                 else:
#                     logs['color'], logs['info'] = 'red', "Not Connected!! Data Query/Path is not updated."
#             self.message['last_log'] = logs
#             self.message['last_log']['color'] = "color:" + self.message['last_log']['color'] + ";"
#             self.logs_update(self, logs)
#
#     def execute_request(self, req, template):
#         print("template :", template, "request :", req)
#         if req != {}:
#             if template == 'manage-data':
#                 self.manage_data_integration(self.check_for_request(req))
#             if template == 'sample-data':
#                 self.create_sample_data(self.check_for_request(req))
#             if template == 'add-data-purchase':
#                 self.data_connections(self.check_for_request(req))
#             if template == 'add-data-action':
#                 self.data_connections(self.check_for_request(req))
#             if template == 'add-data-product':
#                 self.data_connections(self.check_for_request(req))
#             if template == 'add-data-promotion':
#                 self.data_connections(self.check_for_request(req))
#             if template == 'data-execute':
#                 self.data_execute(self.check_for_request(req))
#
#         self.tables = pd.read_sql(self.sqlite_queries['tables'], con)
#         self.message = default_message
#
#     def get_default_es_connection_values(self,
#                                          tables=None,
#                                          active_connections=False,
#                                          hold_connection=False,
#                                          recent_connection=False,
#                                          schedule_connection=False):
#         """
#         tables are the list of tables where you want to fill into values
#         :param tables:
#         :param additional_active_connections:
#         :return:
#         """
#         values = {}
#         for row in range(5):
#             values['row_' + str(row)] = {cols: "...." for cols in self.sqlite_queries['columns']['es_connection']}
#         if active_connections:
#             for row in range(5):
#                 values['row_active_' + str(row)] = {cols: "...." for cols in
#                                                     self.sqlite_queries['columns']['data_connection']}
#             values['active_connections'] = self.message['active_connections']
#
#         if hold_connection:
#             for row in range(5):
#                 values['row_hold_' + str(row)] = {cols: "...." for cols in
#                                                     self.sqlite_queries['columns']['data_connection']}
#         if recent_connection:
#             for row in range(5):
#                 values['row_connect_' + str(row)] = {cols: "...." for cols in
#                                                     self.sqlite_queries['columns']['data_connection']}
#
#         values['orders_data'] = '....'
#         values['downloads_data'] = '....'
#         values['message'] = self.message
#
#         if tables is not None:
#             for row in range(len(tables[0])):
#                 values['row_' + str(row)] = tables[0][row]
#             if active_connections:
#                 try:
#                     for row in range(len(tables[1])):
#                         values['row_active_' + str(row)] = tables[1][row]
#                 except Exception as e:
#                     print(e)
#
#                 try:
#                     values['active_connections'] = tables[1]
#                 except Exception as e:
#                     print(e)
#
#             if hold_connection:
#                 try:
#                     for row in range(len(tables[2])):
#                         values['row_hold_' + str(row)] = tables[2][row]
#                 except Exception as e:
#                     print(e)
#             if recent_connection:
#                 try:
#                     for row in range(len(tables[3])):
#                         values['row_hold_' + str(row)] = tables[3][row]
#                 except Exception as e:
#                     print(e)
#
#         if self.message['orders_data'] != '....':
#             try:
#                 values['orders_data'] = {}
#                 for row in range(len(self.message['orders_data'])):
#                     values['orders_data']['row_orders_data_' + str(row)] = self.message['orders_data'][row]
#             except Exception as e:
#                 print(e)
#
#         if self.message['downloads_data'] != '....':
#             try:
#                 values['downloads_data'] = {}
#                 for row in range(len(self.message['downloads_data'])):
#                     values['downloads_data']['row_downloads_data_' + str(row)] = self.message['downloads_data'][row]
#             except Exception as e:
#                 print(e)
#
#         return values
#
#     def fetch_results(self, template):
#         self.message = default_message
#         # pages manage-data and sample-data of receiving data
#         if template == 'manage-data':
#             if 'es_connection' in list(self.tables['name']):
#                 try:
#                     _table = pd.read_sql("""SELECT * FROM es_connection""", con)
#                     self.table = [_table.reset_index().tail(min(5, len(_table))).to_dict('results')]
#                 except Exception as e:
#                     print("there is no table has been created for now!")
#
#         if template == 'sample-data':
#             if 'es_connection' in list(self.tables['name']):
#                 try:
#                     self.table = [pd.read_sql(""" SELECT e.*
#                                                   FROM es_connection as e
#                                                   LEFT JOIN data_connection as d ON e.tag = d.tag
#                                                   WHERE d.process is NULL and e.status = 'on' """,
#                                               con).reset_index().to_dict('results')]
#                 except Exception as e:
#                     print("there is no table has been created for now!")
#
#             try:
#                 if len(pd.read_sql(
#                         """
#                         SELECT
#                         *
#                         FROM data_connection WHERE process != 'connected' AND dimension = 'sample_data'
#                         """, con)) != 0:
#                         self.table = None
#             except Exception as e:
#                 print(print("there is no 'data_connection' table has been created for now!"))
#
#         # page 'add-data-purchase' of receiving data
#         if template == 'add-data-purchase' or template == 'add-data-action' or \
#            template == 'add-data-product' or template == 'add-data-promotion' or template == 'data-execute':
#             self.active_connections, self.hold_connection, self.recent_connection = True, True, True
#             if 'es_connection' in list(self.tables['name']):
#                 try:
#                     self.table = [pd.read_sql(""" SELECT tag FROM es_connection where status = 'on' """,
#                                               con).reset_index().to_dict('results')]
#                 except Exception as e:
#                     print("there is no table has been created for now!")
#
#                 if 'data_connection' in list(self.tables['name']):
#                     try:
#                         self.table = [pd.read_sql("""
#                                                      SELECT data.id as id,
#                                                             es.tag as tag,
#                                                             data.dimension as dimension,
#                                                             data.orders_data_source_tag,
#                                                             data.downloads_data_source_tag,
#                                                             data.is_action,
#                                                             data.is_product,
#                                                             data.is_promotion
#                                                      FROM es_connection es
#                                                      LEFT JOIN data_connection data
#                                                      ON es.tag = data.tag
#                                                      WHERE status = 'on'
#                                                         AND es.tag not IN (SELECT tag
#                                                                         FROM data_connection
#                                                                         WHERE process = 'connected')
#                                                   """,
#                                                   con).reset_index().to_dict('results')]
#                     except Exception as e:
#                         print("there is no table has been created for now!")
#
#                     try:
#                         self.table += [pd.read_sql(""" SELECT id,
#                                                               tag,
#                                                               dimension,
#                                                               orders_data_source_tag,
#                                                               downloads_data_source_tag,
#                                                               process,
#                                                               is_action,
#                                                               is_product,
#                                                               is_promotion
#                                                         FROM data_connection where process = 'connected' """,
#                                                    con).reset_index().to_dict('results')]
#                     except Exception as e:
#                         print("there is no table has been created for now!")
#                     # check for hold  - edit - add_dimension
#                     try:
#                         self.table += [pd.read_sql(""" SELECT id,
#                                                               tag,
#                                                               dimension,
#                                                               orders_data_source_tag, downloads_data_source_tag, process
#                                                         FROM data_connection
#                                                         WHERE process in ('hold',
#                                                                           'edit',
#                                                                           'add_dimension', 'add_action',
#                                                                           'add_product', 'add_promotion') """,
#                                        con).tail(1).to_dict('results')]
#                     except Exception as e:
#                         print("there is no table has been created for now!")
#
#             if self.check_for_sample_tables():
#                 query_editing_columns = lambda table, type, tag: """ SELECT columns
#                                                                      FROM {}
#                                                                      WHERE data_type = '{}'
#                                                                      AND tag = (SELECT
#                                                                                     {}
#                                                                                 FROM data_connection
#                                                                                 WHERE process in ('hold',
#                                                                                                   'edit',
#                                                                                                   'add_dimension',
#                                                                                                   'add_action',
#                                                                                                   'add_product',
#                                                                                                   'add_promotion') LIMIT 1
#                                                                                 )
#                                                                 """.format(table, type, tag)
#                 main_query = lambda table: """ SELECT * FROM {} """.format(table)
#                 args_creation = lambda type='': {ind: {
#                                                     'table': 'data_columns',
#                                                     'type': type + ind,
#                                                     'tag': ind + '_data_source_tag'} for ind in ['orders', 'downloads'] }
#                 sample_data_tables = lambda type='': {'orders': type + 'orders_sample_data',
#                                                       'downloads': type + 'downloads_sample_data'}
#
#                 args, sample_tables = args_creation(), sample_data_tables()
#                 if template == 'add-data-action':
#                     args, sample_tables = args_creation(type="action_"), sample_data_tables()
#
#                 if template == 'add-data-product':
#                     args, sample_tables = args_creation(type="product_"), sample_data_tables(type="product_")
#
#                 if template == 'add-data-promotion':
#                     args, sample_tables = args_creation(type="promotion_"), sample_data_tables(type="promotion_")
#
#                 try:
#                     _orders_table = pd.read_sql(main_query(sample_tables['orders']), con)
#                     if len(_orders_table) != 0:
#                         self.message['orders_data'] = _orders_table.to_dict('results')
#                     self.message['orders_columns'] = array(list(pd.read_sql(query_editing_columns(**args['orders']),  # 'data_columns', 'orders', 'orders_data_source_tag'
#                                                                 con)['columns'])[0].split("*"))
#                 except Exception as e:
#                     print(e)
#
#                 try:
#                     _downloads_table = pd.read_sql(main_query(sample_tables['downloads']), con)
#                     if len(_downloads_table) != 0:
#                         self.message['downloads_data'] = _downloads_table.to_dict('results')
#                     self.message['downloads_columns'] = array(list(pd.read_sql(query_editing_columns(**args['downloads']),
#                                                                    con)['columns'])[0].split("*"))
#                 except Exception as e:
#                     print(e)
#
#             if template == 'data-execute':
#                 self.recent_connection = True
#                 data_connection, schedules, data_columns = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#                 try:
#                     data_connection = pd.read_sql(""" SELECT * FROM data_connection """, con)
#                 except Exception as e:
#                     print("there is no table has been created for now!")
#                 try:
#                     schedules = pd.read_sql(""" SELECT tag, max_date_of_order_data,
#                                                 time_period, status FROM schedule_data """, con)
#                 except Exception as e:
#                     print("there is no table has been created for now!")
#
#                 try:
#                     data_columns = pd.read_sql(""" SELECT id, tag FROM data_columns_integration """, con)
#                 except Exception as e:
#                     print("there is no table has been created for now!")
#
#                 if len(data_connection) != 0:
#                     for guery in ['orders_data_query_path', 'downloads_data_query_path']:
#                         data_connection[guery] = data_connection[guery].apply(lambda x: '....' if x is None else x)
#                         data_connection[guery] = data_connection[guery].apply(
#                             lambda x: x[:5] + '....' + x[-5:] if len(x) > 40 else x)
#                     data_connection = self.assign_color_for_es_tag(data_connection)
#                     if len(schedules) != 0:
#                         data_connection = pd.merge(data_connection, schedules, on='tag', how='left')  # .fillna('....')
#                     else:
#                         for col in ['max_date_of_order_data', 'time_period']:
#                             data_connection[col] = '....'
#                         data_connection['status'] = 'off'
#                     if len(data_columns) != 0:
#                         renaming = lambda _type: {"tag": _type + "_data_source_tag", "id": _type + '_id'}
#                         data_columns = data_columns.sort_values('id', ascending=True).groupby(
#                             "tag").agg({"id": "max"}).reset_index()
#                         data_connection = pd.merge(data_connection,
#                                                    data_columns.rename(columns=renaming('orders')),
#                                                    on='orders_data_source_tag', how='left').fillna('....')
#                         data_connection = pd.merge(data_connection,
#                                                    data_columns.rename(columns=renaming('downloads')),
#                                                    on='downloads_data_source_tag', how='left').fillna('....')
#                         data_connection['columns_matching'] = data_connection.apply(
#                             lambda row: self.columns_matching_decision(row), axis=1)
#                     else:
#                         data_connection['columns_matching'] = 'not assigned'
#                         data_connection['color'] = '#FFFFFF'
#                     check_for_status = lambda col, x: 'off' if col == 'status' and x == '....' else x
#                     convert_nulls = lambda col, x: check_for_status(col, '....') if x in ['None', None] else x
#                     cols = list(data_connection.columns)
#                     data_connection[cols] = data_connection.apply(
#                         lambda row: pd.Series([convert_nulls(col, row[col]) for col in cols]),axis=1)
#                     schedule_tags = data_connection.query("status != 'on' and process == 'connected'")
#                     if len(schedule_tags) != 0:
#                         self.message['schedule_tags'] = array(
#                                 list(set(list(schedule_tags.query("columns_matching == 'assigned'")['tag'].unique())) -
#                                      set(list(schedule_tags.query("columns_matching != 'assigned'")['tag'].unique()))))
#                     self.message['schedule'] = data_connection.to_dict('results')
#                     self.message['schedule_columns'] = array(schedule_columns)
#                 else:
#                     self.message['schedule_columns'] = self.sqlite_queries['columns']['data_connection'] + \
#                                                        ['max_date_of_order_data', 'time_period', 'status']
#                     self.message['schedule'] = [{cols: "...." for cols in
#                                                  self.sqlite_queries['columns']['data_connection']}]
#
#                 # collect logs
#                 try:
#                    logs = pd.read_sql("SELECT * FROM logs", con)
#                    logs['color'] = logs['color'].apply(lambda x: "color:" + x + ";")
#                    self.message['logs'] = logs.to_dict('results')[-min(10, len(logs)):]
#                 except Exception as e:
#                     print(e)
#
#         values = self.get_default_es_connection_values(tables=self.table,
#                                                        active_connections=self.active_connections,
#                                                        hold_connection=self.hold_connection,
#                                                        recent_connection=self.recent_connection)
#
#         print("VAAAALUEEESSSSS :::", values)
#         return values
#
#
#