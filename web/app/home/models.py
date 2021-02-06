import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from sqlalchemy import create_engine, MetaData
from os.path import abspath, join
from utils import read_yaml
from configs import query_path, default_es_port, default_es_host
import pandas as pd


engine = create_engine('sqlite://///' + join(abspath(""), 'db.sqlite3'), convert_unicode=True, connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


class RouterRequest:
    def __init__(self):
        self.return_values = {}
        self.sqlite_queries = read_yaml(query_path, "queries.yaml")
        self.tables = pd.read_sql(self.sqlite_queries['tables'], con)

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
        _query +=" WHERE  " + condition
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
            con.execute(self.sqlite_queries[table])

    def check_for_request(self, _r):
        _r_updated = {}
        try:
            for i in _r:
                if _r[i] != _r[i] or _r[i] is None or _r[i] in ['', '-', '....']:
                    if 'edit' not in list(_r.keys()):
                        _r_updated[i] = None
                else:
                    if i in ['delete', 'edit']:
                        if _r[i] != 'True':
                            if i == 'edit':
                                _r_updated['recent_tag'] = _r[i]
                            else:
                                _r_updated['tag'] = _r[i]
                        _r_updated[i] = 'True'
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

    def execute_request(self, req, template):
        print("template :", template, "request :", req)
        if req != {}:
            if template == 'manage-data':
                self.manage_data_integration(self.check_for_request(req))

    def fetch_results(self, template):
        values = {}
        print(template)
        if template in ['manage-data', 'sample-data']:
            values = {'row_' + str(row): {cols: "...." for cols in self.sqlite_queries['columns']['es_connection']}
                            for row in range(5)}
            if 'es_connection' in list(self.tables['name']):
                try:
                    table = pd.read_sql(""" SELECT * FROM es_connection """, con).reset_index().tail(5).to_dict('results')
                    for row in range(len(table)):
                        values['row_' + str(row)] = table[row]
                except Exception as e:
                    print("there is no table has been created for now!")
        return values


