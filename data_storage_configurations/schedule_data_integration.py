import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import schedule
import threading
import time
import pandas as pd
from sqlalchemy import create_engine, MetaData
from os.path import abspath, join
from data_storage_configurations.es_create_index import CreateIndex
from data_storage_configurations.sample_data import CreateSampleIndex
from exploratory_analysis import create_exploratory_analysis
from ml_process import create_mls
from utils import current_date_to_day, convert_to_day

engine = create_engine('sqlite://///' + join(abspath(""), "web", 'db.sqlite3'),
                       convert_unicode=True,
                       connect_args={'check_same_thread': False})
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
        self.create_index = CreateIndex(data_connection_structure=data_connection_structure,
                                        data_columns=data_columns, actions=actions)
        self.schedule = True

    def query_schedule_status(self):
        """
        When the process is scheduled for daily, '12 hours', before it starts, checks scheduling is still 'on' or not deleted.
        """
        return pd.read_sql("SELECT * FROM schedule_data ", con)

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
        tag = self.query_schedule_status()
        accept = True
        if tag['time_period'] == '12_hours':
            if tag['max_date_of_order_data'] != 'None':
                time_diff = (convert_to_day(current_date_to_day()) -
                             convert_to_day(tag['max_date_of_order_data'])).total_seconds() / 60 / 60 / 24
                if int(time_diff) == 0:
                    accept = False
        if accept:
            if tag['is_exploratory'] != 'None':
                for _conf in self.ea_connection_structure:
                    create_exploratory_analysis(_conf)
            if tag['is_mlworks'] != 'None':
                for _conf in self.ml_connection_structure:
                    create_mls(_conf)

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
            self.create_index.execute_index()
        else:
            s.do(self.create_index.execute_index())
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
        process.join()
        print("scheduling is triggered!!")




