import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from ml_process.customer_segmentation import CustomerSegmentation
from ml_process.clv_prediction import CLVPrediction
from ml_process.ab_test import ABTests
from ml_process.anomaly_detection import Anomaly

ml_configs = {"date": None,
              'time_period': '6 months',
              "segmentation": {"host": 'localhost', "port": '9200',
                               'download_index': 'downloads', 'order_index': 'orders'},
              "clv_prediction": {"temporary_export_path": None,
                                 "host": 'localhost', "port": '9200',
                                 'download_index': 'downloads', 'order_index': 'orders', 'time_period': 'weekly'},
              "abtest": {"has_product_connection": True,
                         "has_promotion_connection": True, "temporary_export_path": None,
                         "host": 'localhost', "port": '9200', 'download_index': 'downloads', 'order_index': 'orders'},
              "anomaly": {"host": 'localhost', "port": '9200',
                          'download_index': 'downloads', 'order_index': 'orders'},

          }

mls = {'segmentation': CustomerSegmentation,
       'clv_prediction': CLVPrediction,
       'abtest': ABTests,
       'anomaly': Anomaly
       }


def create_ml(configs, ml):
    ea = {a: mls[a](**configs[a]) for a in mls}
    if ml == 'segmentation':
        print("*" * 5, " Customer Segmentation ", "*" * 5)
        ea['segmentation'].execute_customer_segment(start_date=configs['date'])
        del ea['segmentation']
    if ml == 'clv_prediction':
        print("*" * 5, " CLV Prediction ", "*" * 5)
        ea['clv_prediction'].execute_clv(start_date=configs['date'], time_period=configs['time_period'])
        del ea['clv_prediction']
    if ml == 'abtest':
        print("*" * 5, " A/B Test ", "*" * 5)
        ea['abtest'].build_in_tests(date=configs['date'])
        del ea['abtest']
    if ml == 'anomaly':
        print("*" * 5, " Anomaly Detection ", "*" * 5)
        ea['anomaly'].execute_anomaly(date=configs['date'])
        del ea


def query_mls(configs, queries, ea):
    ea = mls[ea](**configs[ea])
    return ea.fetch(**queries)