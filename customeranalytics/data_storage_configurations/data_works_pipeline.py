import sys, os, inspect, logging
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import schedule
import threading
import time
import pandas as pd
from numpy import unique
from sqlalchemy import create_engine, MetaData
from os.path import abspath, join
from flask_login import current_user

try:
    from exploratory_analysis.__init__ import create_exploratory_analysis, create_exploratory_analyse
except Exception as e:
    from customeranalytics.exploratory_analysis import create_exploratory_analysis, create_exploratory_analyse
try:
    from ml_process.__init__ import create_ml
except Exception as e:
    from customeranalytics.ml_process import create_ml

from customeranalytics.utils import current_date_to_day, convert_to_day, abspath_for_sample_data, read_yaml
from customeranalytics.configs import query_path, DATA_WORKS_READABLE_FORM
from customeranalytics.data_storage_configurations.es_create_index import CreateIndex
from customeranalytics.data_storage_configurations.query_es import QueryES


engine = create_engine('sqlite://///' + join(abspath_for_sample_data(), "web", 'db.sqlite3'),
                       convert_unicode=True, connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


class DataPipelines:
    def __init__(self,
                 es_tag,
                 data_connection_structure,
                 ea_connection_structure,
                 ml_connection_structure,
                 data_columns,
                 actions):
        """
        :param es_tag: elasticsearch tag name that is created on web interface 'ElasticSearch Configuration' page
        :param data_connection_structure: check :data_configs above
        :param ea_connection_structure: check :ea_configs above
        :param ml_connection_structure: check :ml_configs above
        :param data_columns: check :data_columns above
        :param actions: whole actions which are stored into the actions table in sqlite
        """
        self.es_tag = es_tag
        self.data_connection_structure = data_connection_structure
        self.ea_connection_structure = ea_connection_structure
        self.ml_connection_structure = ml_connection_structure
        self.actions = actions
        self.data_columns = data_columns
        self.es_con = pd.read_sql("select * from es_connection", con).to_dict('results')[-1]
        self.create_index = CreateIndex(data_connection_structure=data_connection_structure,
                                        data_columns=data_columns, actions=actions)
        self.query_es = QueryES(host=self.es_con['host'], port=self.es_con['port'])
        self.unique_dimensions = []
        self.schedule = True
        self.sqlite_queries = read_yaml(query_path, "queries.yaml")
        self.tables = pd.read_sql(self.sqlite_queries['tables'], con)
        self.separator = lambda dim: [print("*" * 20) for i in range(3)] + [
            print("*" * 10, " ", " DIMENSION : ", dim, "*" * 10)]
        self.info_logs_for_chat = lambda info: {'user': 'info',
                                                'date': str(current_date_to_day())[0:19],
                                                'user_logo': 'info.jpeg',
                                                'chat_type': 'info', 'chart': '',
                                                'general_message': info, 'message': ''}
        self.suc_log_for_ea = """
                               Exploratory Analysis are Created! Check Funnel, Cohort, 
                               Descriptive Stats sections.
                              """
        self.suc_log_for_data_works = lambda x: " {0} is created! Check {0} sections. ".format(
            DATA_WORKS_READABLE_FORM[x])
        self.fail_log_for_data_works = lambda x, e: " {0} is failed! Check {0} sections. - {1}".format(
            DATA_WORKS_READABLE_FORM[x], e)

    def check_for_table_exits(self, table):
        """
        checking sqlite if table is created before. If it not, table is created.
        :params table: checking table name in sqlite
        """

        try:
            if table not in list(self.tables['name']):
                con.execute(self.sqlite_queries[table])
        except Exception as e: print(e)

    def insert_query(self, table, columns, values):
        """
        insert sqlite tables with given table column and values

        :param table: tables to be inserted the row
        :param columns: tables of columns (all columns)
        :param values: values for each column in the table
        """
        values = [values[col] for col in columns]
        _query = "INSERT INTO " + table + " "
        _query += " (" + ", ".join(columns) + ") "
        _query += " VALUES (" + ", ".join([" '{}' ".format(v) for v in values]) + ") "
        _query = _query.replace("\\", "")
        return _query

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
            con.execute(self.insert_query(table='logs',
                                          columns=self.sqlite_queries['columns']['logs'][1:],
                                          values=logs
                                          ))
        except Exception as e: print(e)

        try:
            con.execute(self.insert_query(table='chat', columns=self.sqlite_queries['columns']['chat'][1:],
                                          values=self.info_logs_for_chat(logs['info'])))
        except Exception as e: print(e)

    def info_log_create(self, type, dim=None):
        """
        Logging system on creating E.A. or ML Works. This print processes are also shown at profile.html activities.
        This gives the information of the process are done.
        """
        _info = self.suc_log_for_data_works(type)
        if dim is not None:
            _info += " || dimension : " + dim
        print("-- Process Info --")
        print(_info)
        self.logs_update(logs={"page": "data-execute", "info": _info, "color": "green"})

    def fail_log_create(self, e, type, dim=None):
        """
        Logging system on creating E.A. or ML Works. This print processes are also shown at profile.html activities.
        This gives the information of the process are failed.
        """
        e_str = ''
        if e is not None:
            e_str = str(e)
            e_str = e_str[:min(len(e_str), 100)]
        fail_message = self.fail_log_for_data_works(type, e_str)
        if dim is not None:
            fail_message += " || dimension : " + dim
        print(" FAIL ---- !!!!!!!!")
        print("-- message :", fail_message)
        print(" ----- description ::::::", e_str)
        self.logs_update(logs={"page": "data-execute",
                               "info": fail_message.replace("'", " "),
                               "color": "red"})

    def execute_pipe(self, _conf, execution, type, dim=None):
        try:
            execution(_conf, type)
            self.info_log_create(type=type)
        except Exception as e:
            print(e)
            self.fail_log_create(e=e, type=type, dim=dim)

    def pipe_1(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ml_connection_structure, create_ml, 'clv_prediction')

    def pipe_2(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'rfm', dim=dim)
        self.execute_pipe(ml_connection_structure, create_ml, 'segmentation', dim=dim)

    def pipe_3(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'funnel', dim=dim)
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'cohort', dim=dim)

    def pipe_4(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'stats', dim=dim)
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'churn', dim=dim)

    def pipe_5(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ml_connection_structure, create_ml, 'abtest', dim=dim)
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'products', dim=dim)
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'promotions', dim=dim)

    def pipe_6(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ml_connection_structure, create_ml, 'anomaly', dim=dim)

    def data_work_pipelines_execution(self, ml_connection_structure, ea_connection_structure, dim=None):
        _kwargs = {"ml_connection_structure": ml_connection_structure,
                   'ea_connection_structure': ea_connection_structure, 'dim': dim}
        if dim is None:
            self.pipe_1(**_kwargs)
        self.pipe_2(**_kwargs)
        pipes = [self.pipe_3, self.pipe_4]
        for pipe in pipes:
            process = threading.Thread(target=pipe, kwargs=_kwargs)
            process.daemon = True
            process.start()
        process.join()
        self.pipe_5(**_kwargs)
        self.pipe_6(**_kwargs)
