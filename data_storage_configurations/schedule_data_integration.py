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

engine = create_engine('sqlite://///' + join(abspath(""), "web", 'db.sqlite3'), convert_unicode=True, connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


class Scheduler:
    """

    """
    def __init__(self, es_tag, data_connection_structure):
        """
        query_size: default query size from configs.py.

        :param host: elasticsearch host
        :param port: elasticsearch port
        """
        self.es_tag = es_tag
        self.data_connection_structure = data_connection_structure
        self.create_index = CreateIndex(es_tag=es_tag, data_connection_structure=data_connection_structure)
        self.schedule = True

    def query_schedule_status(self):
        return pd.read_sql("SELECT * FROM schedule_data WHERE tag = '" + self.es_tag + "' ", con)

    def update_schedule_last_time(self):
        con.execute("UPDATE schedule_data  WHERE tag = '" + self.es_tag + "' ")

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

    def execute_schedule(self):

        s = self.create_schedule()
        if s == 'once':
            self.create_index.execute_index()
        else:
            s.do(self.create_index.execute_index())
            self.update_schedule_last_time()
            while self.schedule:
                s.run_pending()
                try:
                    tag = self.query_schedule_status()
                    if list(tag['status'])[0] != 'on':
                        self.schedule = False
                except Exception as e:
                    self.schedule = False
                # time.sleep(721*60)
                time.sleep(2)

    def run_schedule_on_thread(self):
        process = threading.Thread(target=self.execute_schedule)
        process.daemon = True
        process.start()
        process.join()

