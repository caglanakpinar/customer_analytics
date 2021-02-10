import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from sqlalchemy import create_engine, MetaData
from os.path import abspath, join
from utils import read_yaml
from configs import query_path, default_es_port, default_es_host
import pandas as pd

from web.app.base.util import hash_pass


engine = create_engine('sqlite://///' + join(abspath(""), "web", 'db.sqlite3'), convert_unicode=True, connect_args={'check_same_thread': False})
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

    def delete_query(self, table, condition):
        _query = "DELETE FROM " + table
        _query += " WHERE " + condition
        _query = _query.replace("\\", "")
        return _query

    def combine_inserting_columns(self, requests, table):
        for col in self.sqlite_queries['columns'][table]:
            if col in list(requests.keys()):
                if requests[col] is None:
                    requests[col] = 'None'
            else:
                requests[col] = 'None'
        return requests

    def check_for_table_exits(self, table):
        if table not in list(self.tables['name']):
            print()
            con.execute(self.sqlite_queries[table])
            print()

    def data_connections_hold_edit_connection_check(self):
        conns = pd.read_sql(
            """ SELECT id, tag, process FROM data_connection WHERE process in ('hold', 'edit', 'add_dimension') """, con)
        if len(conns) != 0:
            holds, edits = conns.query("process == ('hold', 'add_dimension')"), conns.query("process == 'edit'")
            if len(holds) != 0:
                for _id in list(holds['id'].unique()):
                    con.execute(self.delete_query(table='data_connection', condition=" id = " + str(_id) + " "))

            print()
            if len(edits) != 0:
                for _id in list(edits['id'].unique()):
                    con.execute(self.update_query(table='data_connection',
                                                  columns=["process"],
                                                  values={"process": "connected"},
                                                  condition=" id = '" + str(_id) + "' "))
        print()

    def check_for_request(self, _r):
        _r_updated = {}
        try:
            for i in _r:
                if _r[i] != _r[i] or _r[i] is None or _r[i] in ['', '-', '....']:
                    if 'edit' not in list(_r.keys()):
                        _r_updated[i] = None
                else:
                    if i in ['delete', 'edit', 'connect']:
                        if _r[i] != 'True':
                            if i == 'edit':
                                _r_updated['recent_tag'] = _r[i]
                            else:
                                _r_updated['tag'] = _r[i]
                        _r_updated[i] = 'True'
                    else:
                        if i == 'process':
                            _r_updated[_r[i]] = 'True'
                        else:
                            _r_updated[i] = _r[i]

        except Exception as e:
            print(e)
        return _r_updated

    def manage_data_integration(self, requests):
        print(requests)
        if requests.get('connect', None) == 'True':
            self.check_for_table_exits(table='es_connection')
            requests['port'] = str(default_es_port) if requests['port'] is None else requests['port']
            requests['host'] = str(default_es_host) if requests['host'] is None else requests['host']
            requests['status'] = 'on'

            try:
                con.execute(self.insert_query(table='es_connection',
                                              columns=self.sqlite_queries['columns']['es_connection'][1:],
                                              values=requests))
            except Exception as e:
                print(e)

        if requests.get('edit', None) == 'True':
            try:
                con.execute(self.update_query(table='es_connection',
                                              condition=" tag = '" + requests['recent_tag'] + "' ",
                                              columns=["status", "port", "host", "tag"],
                                              values=requests))
            except Exception as e:
                print(e)

        if requests.get('delete', None) == 'True':
            try:
                print()
                con.execute(self.delete_query(table='es_connection',
                                              condition=" tag = '" + requests['tag'] + "' "))
            except Exception as e:
                print(e)
        self.tables = pd.read_sql(self.sqlite_queries['tables'], con)

    def data_connections(self, requests):
        # for orders index choose ElasticSearch Connection from es_connection table (only status == 'on')
        if requests.get('connect', None) == 'True':
            self.check_for_table_exits(table='data_connection')
            for col in self.sqlite_queries['columns']['data_connection'][1:]:
                if col not in list(requests.keys()):
                    requests[col] = None
            requests['process'] = 'hold'
            self.data_connections_hold_edit_connection_check()
            try:
                con.execute(self.insert_query(table='data_connection',
                                              columns=self.sqlite_queries['columns']['data_connection'][1:],
                                              values=requests))
                print()
            except Exception as e:
                print(e)

        if requests.get('edit', None) == 'True':
            requests['process'] = 'edit'
            self.data_connections_hold_edit_connection_check()
            con.execute(self.update_query(table='data_connection',
                                          condition=" tag = '" + requests['tag'] + "' ",
                                          columns=["process"],
                                          values=requests))

        if requests.get('add_dimension', None) == 'True':
            for col in self.sqlite_queries['columns']['data_connection'][1:]:
                if col not in list(requests.keys()):
                    requests[col] = None
            requests['process'] = 'add_dimension'
            requests['dimension'] = 'True'
            self.data_connections_hold_edit_connection_check()
            con.execute(self.insert_query(table='data_connection',
                                          columns=self.sqlite_queries['columns']['data_connection'][1:],
                                          values=requests))

        if requests.get('delete', None) == 'True':
            con.execute(self.delete_query(table='data_connection', condition=" tag = '" + requests['tag'] + "' "))

        if requests.get('orders_edit', None) == 'True' or requests.get('downloads_edit', None) == 'True':
            source_tag_name = 'orders_data_source_tag'
            if requests.get('orders_edit', None) == 'True':
                source_tag_name = 'downloads_data_source_tag'

            tags = pd.read_sql(
                """
                SELECT
                id, tag, process, """ + source_tag_name +
                """
                FROM data_connection WHERE process in ('hold', 'edit', 'add_dimension') AND dimension != 'sample_data'
                """, con).tail(1)
            id, process = list(tags['id'])[0], list(tags['process'])[0]

            if list(tags[source_tag_name])[0] not in ['', 'None', None, 'Null']:
                requests['process'] = 'connected'
            else:
                requests['process'] = process

            _columns = list(set(list(requests.keys())) & set(self.sqlite_queries['columns']['data_connection'][1:]))

            try:
                con.execute(self.update_query(table='data_connection',
                                              condition=" id = " + str(id) + " ",
                                              columns=_columns,
                                              values=requests))
            except Exception as e:
                print(e)
        self.tables = pd.read_sql(self.sqlite_queries['tables'], con)

    def create_sample_data(self, requests):
        # for orders index choose ElasticSearch Connection from es_connection table (only status == 'on')
        if requests.get('connect', None) == 'True':
            self.check_for_table_exits(table='data_connection')
            for col in self.sqlite_queries['columns']['data_connection'][1:]:
                if col not in list(requests.keys()):
                    requests[col] = None
            requests['process'] = 'connected'
            requests['dimension'] = 'sample_data'
            try:
                con.execute(self.insert_query(table='data_connection',
                                              columns=self.sqlite_queries['columns']['data_connection'][1:],
                                              values=requests))
                print()
            except Exception as e:
                print(e)

    def execute_request(self, req, template):
        print("template :", template, "request :", req)
        if req != {}:
            if template == 'manage-data':
                self.manage_data_integration(self.check_for_request(req))
            if template == 'add-data-purchase':
                self.data_connections(self.check_for_request(req))
            if template == 'sample-data':
                self.create_sample_data(self.check_for_request(req))

    def get_default_es_connection_values(self,
                                         tables=None,
                                         active_connections=False,
                                         hold_connection=False,
                                         recent_connection=False):
        """
        tables are the list of tables where you want to fill into values
        :param tables:
        :param additional_active_connections:
        :return:
        """
        values = {}
        for row in range(5):
            values['row_' + str(row)] = {cols: "...." for cols in self.sqlite_queries['columns']['es_connection']}
        if active_connections:
            for row in range(5):
                values['row_active_' + str(row)] = {cols: "...." for cols in
                                                    self.sqlite_queries['columns']['data_connection']}
        if hold_connection:
            for row in range(5):
                values['row_hold_' + str(row)] = {cols: "...." for cols in
                                                    self.sqlite_queries['columns']['data_connection']}
        if recent_connection:
            for row in range(5):
                values['row_connect_' + str(row)] = {cols: "...." for cols in
                                                    self.sqlite_queries['columns']['data_connection']}

        if tables is not None:
            for row in range(len(tables[0])):
                values['row_' + str(row)] = tables[0][row]
            if active_connections:
                try:
                    for row in range(len(tables[1])):
                        values['row_active_' + str(row)] = tables[1][row]
                except Exception as e:
                    print(e)
            if hold_connection:
                try:
                    for row in range(len(tables[2])):
                        values['row_hold_' + str(row)] = tables[2][row]
                except Exception as e:
                    print(e)
            if recent_connection:
                try:
                    for row in range(len(tables[3])):
                        values['row_hold_' + str(row)] = tables[3][row]
                except Exception as e:
                    print(e)
        return values

    def fetch_results(self, template):
        # pages manage-data and sample-data of receiving data
        if template == 'manage-data':
            if 'es_connection' in list(self.tables['name']):
                try:
                    self.table = [pd.read_sql(""" SELECT * FROM es_connection """,
                                              con).reset_index().tail(5).to_dict('results')]
                except Exception as e:
                    print("there is no table has been created for now!")
        if template == 'sample-data':
            if template == 'manage-data':
                if 'es_connection' in list(self.tables['name']):
                    try:
                        self.table = [pd.read_sql(""" SELECT * 
                                                      FROM es_connection as e 
                                                      LEFT JOIN data_connection as d ON e.tag = d.tag 
                                                      WHERE process != 'connected' """,
                                                  con).reset_index().tail(5).to_dict('results')]
                    except Exception as e:
                        print("there is no table has been created for now!")

            if len(pd.read_sql(
                """
                SELECT
                *
                FROM data_connection WHERE process != 'connected' AND dimension = 'sample_data'     
                """, con)) != 0:
                self.table = None


        # page 'add-data-purchase' of receiving data
        if template == 'add-data-purchase':
            self.active_connections, self.hold_connection, self.recent_connection = True, True, True
            if 'es_connection' in list(self.tables['name']):
                try:
                    self.table = [pd.read_sql(""" SELECT tag FROM es_connection where status = 'on' """,
                                              con).reset_index().tail(5).to_dict('results')]
                except Exception as e:
                    print("there is no table has been created for now!")

                if 'data_connection' in list(self.tables['name']):
                    try:
                        self.table = [pd.read_sql(""" 
                                                     SELECT tag 
                                                     FROM es_connection 
                                                     WHERE status = 'on' 
                                                        AND tag not IN (SELECT tag 
                                                                        FROM data_connection 
                                                                        WHERE process = 'connected') 
                                                  """,
                                                  con).reset_index().tail(5).to_dict('results')]
                    except Exception as e:
                        print("there is no table has been created for now!")

                    try:
                        self.table += [pd.read_sql(""" SELECT tag FROM data_connection where process = 'connected' """,
                                                   con).reset_index().tail(5).to_dict('results')]
                    except Exception as e:
                        print("there is no table has been created for now!")

                    # check for hold  - edit - add_dimension
                    try:
                        self.table += [pd.read_sql(""" SELECT tag, process FROM data_connection 
                                                       WHERE process in ('hold', 'edit', 'add_dimension') """,
                                       con).tail(1).to_dict('results')]
                    except Exception as e:
                        print("there is no table has been created for now!")

        values = self.get_default_es_connection_values(tables=self.table,
                                                       active_connections=self.active_connections,
                                                       hold_connection=self.hold_connection,
                                                       recent_connection=self.recent_connection)

        print("VAAAALUEEESSSSS :::", values)

        return values


