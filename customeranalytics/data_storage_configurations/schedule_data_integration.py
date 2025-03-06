import schedule
import threading
import time

from customeranalytics.data_storage_configurations.data_works_pipeline import DataPipelines


class Scheduler(DataPipelines):
    """
    It allows us to schedule data storage process, triggering Exploratory Analysis and
    Ml Creation Processes jobs sequentially. this will work on a thread that goes on as a background process.
    Once you have shot down the platform, the thread will be killed. In order to cancel the ongoing process,
    Delete schedule job from web interface from 'Schedule Data Process' page.
    There are 4 options for scheduling;
        - Once; Only once, it collects data store into the orders and downloads indexes,
          Exploratory Analysis and ML Processes Creation.
        - Weekly; every Mondays, fetching data and storing it into the indexes.
        - Daily; daily fetching data and storing it into the indexes.

    """
    def __init__(self):
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



        :param data_connection_structure: check :data_configs above
        :param ea_connection_structure: check :ea_configs above
        :param ml_connection_structure: check :ml_configs above
        :param data_columns: check :data_columns above
        :param actions: whole actions which are stored into the actions table in sqlite
        """
        super().__init__()
        self.unique_dimensions = []
        self.schedule = True

    @staticmethod
    def separator(dim):
        return [print("*" * 20) for i in range(3)] + [print("*"*10, " "," DIMENSION : ",dim, "*"*10)]


    def query_schedule_status(self):
        """
        When the process is scheduled for daily, 'weekly',
        before it starts, checks scheduling is still 'on' or not deleted.
        """
        return self.connection_conf.schedule_data

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

    def create_schedule(self):
        tag = self.query_schedule_status()
        time_period = list(tag['time_period'])[0]
        if time_period == 'daily':
            return schedule.every().day.at("00:00")
        if time_period == 'weekly':
            return schedule.every().monday.at("00:00")
        if time_period == 'once':
            return 'once'


    def execute_schedule(self, job: callable):
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
        if s == 'once':  # no need to schedule for once triggering process
            print(self.es_tag, " - triggered for once !!!")
            job()
        else:
            s.do(job)
            while self.schedule:
                schedule.run_pending()
                try:
                    tag = self.query_schedule_status()  # when schedule record is removed or it is canceled, returns 0
                    if len(tag) == 0:
                        print("schedule is cancelled")
                        self.schedule = False
                except Exception as e:
                    self.schedule = False
                time.sleep(100)
                print("waiting ....")

    def run_schedule_on_thread(self, function, args=None):
        """
        allows us to create Orders and Downloads Indexes on the thread.
        """
        process = threading.Thread(target=function, kwargs={} if args is None else args)
        process.daemon = True
        process.start()
        print("scheduling is triggered!!")




