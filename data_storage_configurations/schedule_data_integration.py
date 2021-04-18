import sys, os, inspect
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
from data_storage_configurations.es_create_index import CreateIndex
from data_storage_configurations.query_es import QueryES
from flask_login import current_user
try:
    from exploratory_analysis.__init__ import create_exploratory_analysis
except Exception as e:
    from exploratory_analysis import create_exploratory_analysis
try:
    from ml_process.__init__ import create_mls
except Exception as e:
    from ml_process import create_mls

from utils import current_date_to_day, convert_to_day, abspath_for_sample_data, read_yaml
from configs import query_path

engine = create_engine('sqlite://///' + join(abspath_for_sample_data(), "web", 'db.sqlite3'),
                       convert_unicode=True, connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


class Scheduler:
    """
    It allows us to schedule data storage process, triggering Exploratory Analysis and
    Ml Creation Processes jobs sequentially. this will work on a thread that goes on as a background process.
    Once you have shot down the platform, the thread will be killed. In order to cancel the ongoing process,
    Delete schedule job from web interface from 'Schedule Data Process' page.
    There are 4 options for scheduling;
        - Once; Only once, it collects data store into the orders and downloads indexes,
          Exploratory Analysis and ML Processes Creation.
        - 12 Hours, Every each 12 hours scheduling queries the data sources and
          fetches the data and stores data into the indexes.
          However, it waits for Exploratory Analysis and ML Processes Creation for the end of the day.
          Exploratory Analysis and ML Processes are triggered daily.
        - Daily; daily fetching data and storing it into the indexes.

    """
    def __init__(self,
                 es_tag,
                 data_connection_structure,
                 ea_connection_structure,
                 ml_connection_structure,
                 data_columns,
                 actions):
        """
        Example of data_connection_structure;
            For more details pls check 'create_data_access_parameters', 'get_data_connection_arguments'  in __init__.py.

            data_configs = {'orders': {'data_source': 'postgresql', 'data_query_path': 'select  * ...', ...},
                            'downloads': {'data_source': 'postgresql', 'data_query_path': 'select  * ...', ...},
                            'products': {'data_source': 'postgresql', 'data_query_path': 'select  * ...', ...}
                            }

        Example of ea_connection_structure;
            For more details pls check 'get_ea_and_ml_config'  in __init__.py.

            ea_configs = {"date": None,
                          "funnel": {"actions": ["download", "signup"],
                                     "purchase_actions": ["has_basket", "order_screen"],
                                     "host": 'localhost',
                                     "port": '9200',
                                     'download_index': 'downloads',
                                     'order_index': 'orders'},
                          "cohort": {"has_download": True, "host": 'localhost', "port": '9200'},
                          "products": {"has_download": True, "host": 'localhost', "port": '9200'},
                          "rfm": {"host": 'localhost', "port": '9200',
                                  'download_index': 'downloads', 'order_index': 'orders'},
                          "stats": {"host": 'localhost', "port": '9200',
                                    'download_index': 'downloads', 'order_index': 'orders'}
                          }


        Example of ea_connection_structure;
            For more details pls check 'get_ea_and_ml_config'  in __init__.py.

            ml_configs = {"date": None,
                          "segmentation": {"host": 'localhost', "port": '9200',
                                           'download_index': 'downloads', 'order_index': 'orders'},
                          "clv_prediction": {"temporary_export_path": None,
                                             "host": 'localhost', "port": '9200',
                                             'download_index': 'downloads', 'order_index': 'orders'},
                          "abtest": {"temporary_export_path": None,
                                     "host": 'localhost', "port": '9200',
                                     'download_index': 'downloads', 'order_index': 'orders'}
                         }

        Example of data_columns;
            For more details pls check 'get_data_connection_arguments'  in __init__.py.

            |order_id |	client	| session_start_date |	payment_amount	| ....| category	promotion_id | ...
            --------------------------------------------------------------------------------------------------
            |order_id |	client	| session_start_date |	payment_amount	| ....| category	promotion_id | ...



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
        self.es_con = pd.read_sql("select * from es_connection", con).to_dict('results')[-1]
        self.create_index = CreateIndex(data_connection_structure=data_connection_structure,
                                        data_columns=data_columns, actions=actions)
        self.query_es = QueryES(host=self.es_con['host'], port=self.es_con['port'])
        self.unique_dimensions = []
        self.schedule = True
        self.sqlite_queries = read_yaml(query_path, "queries.yaml")
        self.tables = pd.read_sql(self.sqlite_queries['tables'], con)
        self.suc_log_for_ea = """
                               Exploratory Analysis are Created! Check Funnel, Cohort, 
                               Descriptive Stats sections.
                              """
        self.suc_log_for_ml = """
                               ML Works are Created! Check CLV Prediction, AB Test, Anomaly,
                               Customer Segmentation sections.
                              """
        self.fail_log_for_ea = " Exploratory Analysis Creation is failed! - "
        self.fail_log_for_ml = " ML Works Creation is failed! - "

    def query_schedule_status(self):
        """
        When the process is scheduled for daily, '12 hours', before it starts, checks scheduling is still 'on' or not deleted.
        """
        return pd.read_sql("SELECT * FROM schedule_data ", con)

    def collect_dimensions_for_data_works(self):
        """

        """
        _res = self.query_es.es.search(index='orders', body={"size": self.query_es.query_size, "from": 0, '_source': False,
                                                             "fields": ["dimension"]})['hits']['hits']
        _res = [r['fields']['dimension'][0] for r in _res]
        self.unique_dimensions = unique(_res).tolist()

    def check_for_table_exits(self, table, query=None):
        try:
            if table not in list(self.tables['name']):
                if query is None:
                    con.execute(self.sqlite_queries[table])
                else:
                    con.execute(query)
        except Exception as e:
            print(e)

    def insert_query(self, table, columns, values):
        values = [values[col] for col in columns]
        _query = "INSERT INTO " + table + " "
        _query += " (" + ", ".join(columns) + ") "
        _query += " VALUES (" + ", ".join([" '{}' ".format(v) for v in values]) + ") "
        _query = _query.replace("\\", "")
        return _query

    def logs_update(self, logs):
        try:
            self.check_for_table_exits(table='logs')
        except Exception as e:
            print(e)
        try:
            logs['login_user'] = current_user
            logs['log_time'] = str(current_date_to_day())[0:19]
            con.execute(self.insert_query(table='logs',
                                          columns=self.sqlite_queries['columns']['logs'][1:],
                                          values=logs
                                          ))
        except Exception as e:
            print(e)

    def check_number_of_days(self):
        """

        """

    def create_schedule(self):
        tag = self.query_schedule_status()
        time_period = list(tag['time_period'])[0]
        if time_period == 'daily':
            return schedule.every().day.at("00:00")
        if time_period == 'weekly':
            return schedule.every().monday.at("00:00")
        if time_period == '12_hours':
            return schedule.every(720).minutes
        if time_period == 'once':
            return 'once'

    def data_works(self):
        """
        Execute Exploratory Analysis and Machine Learning Works which are implemented in the platform.
        This process is optional on the web interface so, it also checks 'is_mlworks' and 'is_exploratory'.
            Exploratory Analysis;
                - Funnels
                - Cohorts
                - Descriptive Statistics
                - RFM
                - Product Analytics
                - Promotion Analytics
            Machine Learning;
                - Customer Segmentation
                - CLV Prediction
                - A/B Test
                - Anomaly Detection

        These jobs are created per main and dimensional models individually but, are stored in the 'reports' index.
        If the scheduled time period is '12 Hours', the process will wait for day change.
        """
        tag = self.query_schedule_status().to_dict('results')[0]
        accept = True
        if tag['time_period'] == '12_hours':
            if tag['max_date_of_order_data'] != 'None':
                time_diff = (convert_to_day(current_date_to_day()) -
                             convert_to_day(tag['max_date_of_order_data'])).total_seconds() / 60 / 60 / 24
                if int(time_diff) == 0:
                    accept = False
        if accept:
            try:
                print("Exploratory Analysis are initialized !")
                print("arguments : ")
                print(self.ea_connection_structure)
                create_exploratory_analysis(self.ea_connection_structure)
                self.logs_update(logs={"page": "data-execute", "info": self.suc_log_for_ea, "color": "green"})
            except Exception as e:
                self.logs_update(logs={"page": "data-execute",
                                       "info": self.fail_log_for_ea + str(e)[:max(len(str(e)), 100)].replace("'", " "),
                                       "color": "red"})
            try:
                print("ML Works are initialized !")
                print("arguments : ")
                print(self.ml_connection_structure)
                create_mls(self.ml_connection_structure)
                self.logs_update(logs={"page": "data-execute", "info": self.suc_log_for_ml, "color": "green"})
            except Exception as e:
                self.logs_update(logs={"page": "data-execute",
                                       "info": self.fail_log_for_ml + str(e)[:max(len(str(e)), 100)].replace("'", " "),
                                       "color": "red"})

            try:
                self.collect_dimensions_for_data_works()
                if len(self.unique_dimensions) > 1:
                    print("Execute Ml Works and Exploratory Analysis for the Dimensions!")
                    for dim in self.unique_dimensions:
                        print("*" * 20)
                        print("*"*10, " ", " Dimension Name : ", dim, "*"*10)
                        for i in [{"executor": create_exploratory_analysis,
                                   "config": self.ea_connection_structure},
                                  {"executor": create_mls,
                                   "config": self.ml_connection_structure}]:
                            for ea in i['config']:
                                if ea not in ['date', 'time_period']:
                                    i['config'][ea]['order_index'], i['config'][ea]['download_index'] = dim, dim
                            try:
                                print("configs :")
                                print(i['config'])
                                i['executor'](i['config'])
                                _info = self.suc_log_for_ml if i['executor'] == create_mls else self.suc_log_for_ea
                                _info += " || dimension : " + dim
                                self.logs_update(logs={"page": "data-execute", "info": _info, "color": "green"})
                            except Exception as e:
                                fail_message = self.fail_log_for_ml if i['executor'] == create_mls else self.fail_log_for_ea
                                fail_message += e[:max(len(e), 100)]
                                self.logs_update(logs={"page": "data-execute", "info": fail_message, "color": "red"})
            except Exception as e:
                print(e)

    def jobs(self):
        """
        Sequentially, this is the process of scheduling which is starting with data insert into the indexes.
        The it continues with data works which includes ml works and exploratory analsis.
        """
        self.create_index.execute_index()
        self.create_index = None
        print("Orders and Downloads Indexes Creation processes are ended!")
        self.data_works()
        print("Exploratory Analysis and ML Works Creation processes are ended!")

    def execute_schedule(self):
        """
        ElasticSearch Service of Data Scheduling;
            1. This process works on the thread, sequentially.
            2. Checks time period; if the time period is 'once' it is triggered without scheduling.
            3. After lunched for the scheduling (table name; schedule_data),
              'max_date_of_order_data' will be updated on (check update_schedule_last_time)
            4. If scheduling is deleted from the web-interface,
               status columns on update_schedule_last_time will be updated or the whole row will be deleted.
               In that case, the scheduling process will be killed.
               During the whole process, It also checks the 'status' column on the 'schedule_data' table.
        """
        s = self.create_schedule()
        if s == 'once':
            print(self.es_tag, " - triggered for once !!!")
            self.jobs()
        else:
            s.do(self.jobs)
            while self.schedule:
                s.run_pending()
                try:
                    tag = self.query_schedule_status()
                    if len(tag) == 0:
                        self.schedule = False
                except Exception as e:
                    self.schedule = False
                # time.sleep(721*60)
                time.sleep(2)

    def run_schedule_on_thread(self, function, args=None):
        """
        allows us to create Orders and Downloads Indexes on the thread.
        """
        process = threading.Thread(target=function, kwargs={} if args is None else args)
        process.daemon = True
        process.start()
        # process.join()
        print("scheduling is triggered!!")




