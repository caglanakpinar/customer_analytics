import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from ml_process.customer_segmentation import CustomerSegmentation
from ml_process.clv_prediction import CLVPrediction
from ml_process.ab_test import ABTests
from ml_process.anomaly_detection import Anomaly

ml_configs = {"date": None,
              'time_period': 'weekly',
              "segmentation": {"host": 'localhost', "port": '9200',
                               'download_index': 'downloads', 'order_index': 'orders'},
              "clv_prediction": {"temporary_export_path": None,
                                 "host": 'localhost', "port": '9200',
                                 'download_index': 'downloads', 'order_index': 'orders', 'time_period': 'weekly'},
              "abtest": {"temporary_export_path": None,
                         "host": 'localhost', "port": '9200', 'download_index': 'downloads', 'order_index': 'orders'}
          }

mls = {'segmentation': CustomerSegmentation,
       'clv_prediction': CLVPrediction,
       'abtest': ABTests,
       'anomaly': Anomaly
       }


def create_mls(configs):
    ea = {a: mls[a](**configs[a]) for a in mls}
    ea['segmentation'].execute_customer_segment(start_date=configs['date'])
    ea['clv_prediction'].execute_clv(start_date=configs['date'], time_period=configs['time_period'])
    ea['abtest'].build_in_tests(date=configs['date'])
    ea['anomaly'].execute_anomaly(date=configs['date'])


def query_mls(configs, queries, ea):
    ea = mls[ea](**configs[ea])
    return ea.fetch(**queries)