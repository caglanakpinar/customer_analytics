import logging


import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from sqlalchemy import create_engine, MetaData
from os.path import abspath, join
import pandas as pd
from flask_login import current_user


from customeranalytics.utils import read_yaml, current_date_to_day, abspath_for_sample_data

engine = create_engine('sqlite://///' + join(abspath_for_sample_data(), "web", 'db.sqlite3'), convert_unicode=True,
                       connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


class LogsBasicConfeger:
        try: directory = pd.read_sql("select * from es_connection", con).to_dict('results')[-1]['directory']
        except: directory = currentdir

        try: user = current_user['email']
        except: user = 'logs'

        file_path = join(currentdir, "logs.log")
        logging.basicConfig(filename=file_path,
                            level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(message)s')



