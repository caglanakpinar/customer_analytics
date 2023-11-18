import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.ml_process.customer_segmentation import CustomerSegmentation
from customeranalytics.ml_process.clv_prediction import CLVPrediction
from customeranalytics.ml_process.ab_test import ABTests
from customeranalytics.ml_process.anomaly_detection import Anomaly
from customeranalytics.ml_process.delivery_analytics import DeliveryAnalytics

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
              "delivery_anomaly": {"host": 'localhost', "port": '9200',
                                   'download_index': 'downloads', 'order_index': 'orders', 
                                   "temporary_export_path": None, 'has_delivery_connection': True},

          }

mls = {'segmentation': CustomerSegmentation,
       'clv_prediction': CLVPrediction,
       'abtest': ABTests,
       'anomaly': Anomaly,
       'delivery_anomaly': DeliveryAnalytics
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
    if ml == 'delivery_anomaly':
        print("*" * 5, " Delivery Analytics ", "*" * 5)
        ea['delivery_anomaly'].execute_delivery_analysis()
        del ea


def query_mls(configs, queries, ea):
    ea = mls[ea](**configs[ea])
    return ea.fetch(**queries)