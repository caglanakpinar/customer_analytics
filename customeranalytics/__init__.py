import sys, os, inspect
from os.path import join
import subprocess
import pandas as pd
import urllib
import time
from sqlalchemy import create_engine, MetaData
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

try: from utils import abspath_for_sample_data
except: from .utils import abspath_for_sample_data

try: from web.app.home.models import RouterRequest
except: from customeranalytics.web.app.home.models import RouterRequest

try: from web.app.home.forms import RealData, SampleData
except: from customeranalytics.web.app.home.forms import RealData, SampleData

from customeranalytics.web.config import config_dict

try: from web.config import web_configs
except: from customeranalytics.web.config import web_configs

try: from data_storage_configurations import get_data_connection_arguments, \
    decision_for_product_conn, decision_for_promotion_conn, get_ea_and_ml_config
except: from customeranalytics.data_storage_configurations import get_data_connection_arguments, \
    decision_for_product_conn, decision_for_promotion_conn, get_ea_and_ml_config

try: from .exploratory_analysis import ea_configs
except: from customeranalytics.exploratory_analysis import ea_configs

try: from .exploratory_analysis import ml_configs
except: from customeranalytics.ml_process import ml_configs


engine = create_engine('sqlite://///' + join(abspath_for_sample_data(), "web", 'db.sqlite3'),
                       convert_unicode=True, connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


r = RouterRequest()
reports = RealData()
sample_reports = SampleData()


def url_string(value, res=False):
    if value is not None:
        if res:
            return value.replace("\r", " ").replace("\n", " ").replace(" ", "+")
        else:
            return value.replace("+", " ")
    else:
        return None


def request_url(url):
    try:
        res = urllib.request.urlopen(url)
    except Exception as e:
        print(e)
    time.sleep(2)


def create_user_interface():
    """
    This process triggers web interface
    port: port is stored at web_interface.yaml
    host: host is stored at web_interface.yaml
    """
    path = join(abspath_for_sample_data(), "web", "run.py")
    cmd = "python " + path
    print('http://' + str(web_configs['host']) + ':' + str(web_configs['port']))
    _p = subprocess.Popen(cmd, shell=True)


def kill_user_interface():
    """
    This process kills the thread for web interface
    """
    print('http://' + str(web_configs['host']) + ':' + str(web_configs['port']) + '/shutdown')
    request_url(url='http://' + str(web_configs['host']) + ':' + str(web_configs['port']) + '/shutdown')


def collect_data_source():
    """

            _ds = {'data_source': connection[index + '_data_source_type'],
            'date': date,
            'data_query_path': sqlite_string_converter(connection[index + '_data_query_path'], back_to_normal=True),
            'test': test,
            'config': {'host': connection[index + '_host'],
                       'port': connection[index + '_port'],
                       'password': connection[index + '_password'],
                       'user': connection[index + '_user'], 'db': connection[index + '_db']}}
    """
    columns, data_configs = get_data_connection_arguments()[1:]
    has_product_connection = decision_for_product_conn(data_configs)
    has_promotion_connection = decision_for_promotion_conn(columns)
    _ea_configs, _ml_configs, _actions = get_ea_and_ml_config(ea_configs, ml_configs,
                                                              has_product_connection, has_promotion_connection)

    for ds in data_configs:
        for c in data_configs[ds]['config']:
            if c == 'password':
                data_configs[ds]['config']['password'] = "*****"

    return data_configs


def create_ElasticSearch_connection(port, host, temporary_path):
    """
    ElasticSearch configurations with host and port.
    Another requirement which is temporary path is for importing files such as CLV Prediction model files and
    .csv format files with build_in_reports folder.

    :param port: elasticsearch port
    :param host: elasticsearch host
    :param temporary_path: folder path for importing data into the given directory in .csv format.
    """
    request = {'tag': 'es_con',
               'url': "http://{host}:{port}/".format(**{'host': str(host), 'port': str(port)}),
               "port": str(port), 'host': str(host), 'directory': temporary_path, "connect": 'True'}
    r.manage_data_integration(r.check_for_request(request))


def create_connections(dimension_sessions, dimension_customers,
                       sessions_fields, customer_fields, product_fields,
                       products_connection, customers_connection, sessions_connection,
                       actions_sessions='None', actions_customers='None', promotion_id='None'):
    """

    """
    args = {"sessions": [sessions_fields, "orders", sessions_connection, dimension_sessions, actions_sessions],
            "products": [product_fields, "products", products_connection],
            "customers": [customer_fields, "downloads", customers_connection, dimension_customers, actions_customers]}
    for i in args:
        _fields = args[i][0].split("*")
        _req = {"connect": args[i][1],
                i + "_data_source_tag": i + "_data_source",
                "dimension": args[i][3], "actions": args[i][4]}

        if args[i][0] == 'sessions':
            _req["promotion_id"] = promotion_id

        if args[i][0] in ['sessions', 'customers']:
            _req["actions"] = args[i][4]

        for col in args[i][0]:
            _req[col] = args[i][0][col]
        for col in args[i][2]:
            _req[col] = args[i][0][col]
        r.data_connections(r.check_for_request(_req))


def create_schedule(time_period):
    """

    """
    es_tag = list(pd.read_sql("select tag from es_connection", con)['tag'])[0]
    request = {'schedule': 'True', 'time_period': time_period, "es_tag": es_tag}
    r.data_execute(request)


def collect_report(report_name, date=None, dimension='main'):
    """
    If there is a report need as .csv format.
    """
    reports.fetch_report(report_name, index=dimension, date=date)


def report_names():
    """
    Collect all possible report names. These report names are .csv files at sample_data folder.
    """
    return sample_reports.kpis