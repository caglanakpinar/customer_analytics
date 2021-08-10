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
    decision_for_product_conn, decision_for_promotion_conn, get_ea_and_ml_config, check_elasticsearch
except: from customeranalytics.data_storage_configurations import get_data_connection_arguments, \
    decision_for_product_conn, decision_for_promotion_conn, get_ea_and_ml_config, check_elasticsearch

try: from .exploratory_analysis import ea_configs
except: from customeranalytics.exploratory_analysis import ea_configs

try: from .exploratory_analysis import ml_configs
except: from customeranalytics.ml_process import ml_configs

try: from .configs import none_types, session_columns, customer_columns
except: from customeranalytics.configs import none_types, session_columns, customer_columns


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


def create_connections(customers_connection,
                       sessions_connection,
                       products_connection=None,
                       sessions_fields={},
                       customer_fields={},
                       product_fields={},
                       actions_sessions='None',
                       actions_customers='None',
                       promotion_id='None',
                       dimension_sessions=None):
    """
    This process is for the connecting data sources which are SESSIONS, CUSTOMERS PRODUCTS.
    Each data source has on its own unique form to connect.
    This process checks the connection failure, before store the connection information to the sqlite DB.
    There are 3 main updating processes for the store process;
        1.  db connection; These process included data_source_name, data_source_type, DB_name, DB_user, DB_name,
                           data_source_path/query, etc.
        2. actions; This is for actions both for Sessions and Customer Data Source.
            Example of actions; "has_basket, order_screen". Actions are split with ',' and string format.
        3. column names; data source columns must be matched with ElasticSearch fields
                         for each data source (Session, Customers, products) individually. column names must be string format.
            Example of columns;
                *** SESSION COLUMNS ***
                sessions_fields = {'order_id': unique ID for each session (purchased/non-purchased sessions),
                                   'client': ID for client this column can connect with client ID (client_2) or customer_fields,
                                   'session_start_date': eligible date format (yyyy-mm-dd hh:mm),
                                   'date': eligible date format (yyyy-mm-dd hh:mm), ,
                                   'payment_amount': value of purchase (float/int). If it not purchased, please assign None, Null '-',
                                   'discount_amount': discount of purchase (float/int). If it not purchased, please assign None, Null '-',
                                   'has_purchased':
                                                True/False. If it is True, session ends with purchased.
                                                            If it is False, session ends with non-purchased.
                ** optional columns;  date, discount_amount

                *** CUSTOMERS COLUMNS ***
                customer_fields = {'client_2': unique ID for client this column can connect with client ID (client) or session_fields,
                                   'download_date': eligible date format (yyyy-mm-dd hh:mm).
                                                    This date can be any date wich customers first appear at the business.
                                   'signup_date': eligible date format (yyyy-mm-dd hh:mm).
                                                  First event of timestamp after the download_Date per customer
                                   }
                ** optional columns;  signup_date

                *** PRODUCT COLUMNS ***
                product_fields = {'order_id': Order ID for each session which has the created basket,
                                              This column is eligible to merge with Order Id column at session fields.
                                  'product': product Id or name. Make sure it is easy to read from charts
                                  'price': price per each product,
                                  'category': product of category
                                  }


                Products and sessions data sets are stored into the orders Index at ElasticSearch.
                Customer data sets are stored into the customers Index at ElasticSearch.
        4. promotions; ** optional ** Each order might have promotion ID which is an indicator of the organic/inorganic order.
                       It would be more efficient, if both promotion ID and discount amount is assigned at the same data set.

        5. dimension; ** optional ** While you need to split the data set you need to assign dimensional column to the session data set.
                      This process will be directly added to ElasticSearch orders (sessions) index and
                      you may choose dimension filter in the dashboards.

    :param customers_connection: dictionary with data_source, data_query_path, host, port, password, user, db
    :param sessions_connection: dictionary with data_source, data_query_path, host, port, password, user, db
    :param products_connection: dictionary with data_source, data_query_path, host, port, password, user, db
    :param sessions_fields: dictionary with order_id, client, session_start_date, date, payment_amount, discount_amount, has_purchased
    :param customer_fields: client_2, download_date, signup_date
    :param product_fields: order_id, product, price, category
    :param actions_sessions: string with comma separated for sessions
    :param actions_customers: string with comma separated for customers
    :param promotion_id: string column name for promotions
    :param dimension_sessions: string column name for dimensions
    """

    args = {"sessions": [sessions_fields, "orders", sessions_connection, dimension_sessions, actions_sessions],
            "products": [product_fields, "products", products_connection],
            "customers": [customer_fields, "downloads", customers_connection, "", actions_customers]}

    # check it is eligible to insert data source
    ready_for_insert = True
    return_message = ""
    session_column_need = session_columns - set(args['sessions'][0].keys())
    customer_column_need = customer_columns - set(args['customers'][0].keys())

    # check data sources (sessions/customers) have data_source_type and data_source_type
    if args['sessions'][2].get('data_source_type', None) in none_types or \
        args['customers'][2].get('data_query_path', None) in none_types or \
            args['sessions'][2].get('data_source_type', None) in none_types or \
            args['customers'][2].get('data_query_path', None) in none_types:
        ready_for_insert = False
        return_message += """
        - Please make sure customers and sessions connections have both data_source_type and data_source_type. \n
        """
    # check sessions data source has matched data columns
    if len(session_column_need) != 0:
        ready_for_insert = False
        return_message += """
        - Please make sure all session columns are assigned to the sessions_fields parameter. Here are the forgotten ones;  
        """ + ", ".join(session_column_need) + " \n "
    # check sessions data source has matched data columns
    if len(customer_column_need) != 0:
        ready_for_insert = False
        return_message += """
        - Please make sure all customer columns are assigned to the customer_fields parameter. Here are the forgotten ones;  
        """ + ", ".join(customer_column_need) + " \n "

    try:
        es_con = pd.read_sql(""" SELECT *  FROM es_connection """, con).to_dict('results')[0]
        connection, message = check_elasticsearch(es_con['port'], es_con['host'], es_con['directory'])
    except:
        connection, message = False, """
        ElasticSearch Connection Failed Check ES port/host or temporary path or Add new ElasticSearch connection
        """

    if not connection:
        ready_for_insert = False
        return_message += message + " \n "

    if ready_for_insert:
        for i in args:
            # _fields = args[i][0].split("*")
            if i in ['sessions', 'customers']:
                _req = {"connect": args[i][1],
                        args[i][1] + "_data_source_tag": i + "_data_source",
                        "dimension": args[i][3], "actions": args[i][4]}
            else:
                _req = {"connect": args[i][1], args[i][1] + "_data_source_tag": i + "_data_source"}

            if i == 'sessions':
                _req["promotion_id"] = promotion_id

            if i in ['sessions', 'customers']:
                _req["actions"] = args[i][4]

            for col in args[i][0]:
                _req[col] = args[i][0][col]

            if args[i][2] is not None:
                for col in args[i][2]:
                    _req[args[i][1] + '_' + col] = args[i][2][col]
            r.data_connections(r.check_for_request(_req))
    else: print(return_message)


def create_schedule(time_period):
    """
    There are 3 options for scheduling;
        - Once; it is started for only once that triggers all processes but it is never triggered again.
        - Daily; Every day at 00:00, it is started for all processes includes exploratory analyisis, Ml Processes,
                 Data Transferring to the ElasticSearch.
        - 12 Hours; Every 12 hours data transferring is going to be started. And every 24 hours all processes includes
                    exploratory analysis, Ml Processes, Data Transferring to the ElasticSearch are triggered together.

    """
    delete_schedule()
    es_tag = list(pd.read_sql("select tag from es_connection", con)['tag'])[0]
    request = {'schedule': 'True', 'time_period': time_period, "es_tag": es_tag}
    r.data_execute(request)


def delete_schedule():
    """
    In order to stop schedule, you need to delete schedule process.
    Schedule with time_period='once', can only stop with ending running process.
    """
    try:
        con.execute("delete from schedule_data where id = 1")
    except Exception as e:
        print(e)


def collect_report(report_name, date=None, dimension='main'):
    """
    If there is a report need as .csv format.
    """
    report = reports.fetch_report(report_name, index=dimension, date=date)
    if report is False:
        print("reports is not created")
        return None
    else: return report


def report_names():
    """
    Collect all possible report names. These report names are .csv files at sample_data folder.
    """
    return list(sample_reports.kpis.keys())