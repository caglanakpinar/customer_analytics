import sys, os, inspect
from os.path import join, dirname, abspath
from sqlalchemy import create_engine, MetaData
from pandas import read_sql, DataFrame
import subprocess
import requests
from dateutil.parser import parse

currentdir = dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = dirname(currentdir)
sys.path.insert(0, parentdir)

from data_storage_configurations.data_access import GetData
from data_storage_configurations.sample_data import CreateSampleIndex
from data_storage_configurations.es_create_index import CreateIndex
from data_storage_configurations.schedule_data_integration import Scheduler
from configs import elasticsearch_connection_refused_comment, query_path, acception_column_count
from utils import read_yaml

engine = create_engine('sqlite://///' + join(abspath(""), "web", 'db.sqlite3'), convert_unicode=True, connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()
sample_data_columns = read_yaml(query_path, "queries.yaml")['columns']


data_config = {'orders': {'main':
                              {'connection': {},
                               'action': [],
                               'product': [],
                               'promotion': []
                               },
                          'dimensions': [{'connection': {},
                                          'action': [],
                                          'product': [],
                                          'promotion': []
                                          }
                                         ]
                          },
               'downloads': {'main': {'connection': {},
                                      'action': []
                                      },
                             'dimensions': [{'connection': {},
                                             'action': []
                                             }
                                            ]
                             }
               }


def create_connection_columns(index='orders'):
    return [index + i for i in ['_data_source_tag', '_data_source_type',
                                '_data_query_path', '_password', '_user', '_port', '_host', '_db']]


def create_data_access_parameters(connection, index='orders', date=None, test=False, columns=None):
    _ds = {'data_source': connection[index + '_data_source_type'],
            'date': date,
            'data_query_path': connection[index + '_data_query_path'],
            'test': test,
            'config': {'host': connection[index + '_host'],
                       'port': connection[index + '_port'],
                       'password': connection[index + '_password'],
                       'user': connection[index + '_user'], 'db': connection[index + '_db']}
            }


def create_date_structure(es_tag_connections, data_config):
    for index in [("orders_data_source_tag", "orders"), ("downloads_data_source_tag", "downloads")]:
        conns = es_tag_connections.query(index[0] + " == " + index[0])
        _columns = create_connection_columns(index=index[1])
        _query = " and ".join(
            " == ".join(zip(["is_action", "dimension", "is_product", "is_promotion"], ["'None'"] * 4)))
        _query_dimension = _query + "process == 'add_dimension'"
        main = conns.query(_query)[_columns]
        dimensions = conns.query(_query_dimension)[_columns]
        actions = conns.query("is_action == 'True'")[_columns]
        products = conns.query("is_product == 'True'")[_columns]
        promotions = conns.query("is_promotion == 'True'")[_columns]

        for type in [('main', 0)] + list(zip(['dimension'] * len(dimensions), list(range(len(dimensions))))):
            _data = main.to_dict('results')[0] if type[1] == 'main' else dimensions.to_dict('results')[type[1]]
            data_config[index[1]][type[0]]['connection'] = create_data_access_parameters(_data,
                                                                                      index=index[1], date=None,
                                                                                      test=False)
            datasets = [('actions', actions.query(_query))]
            if index[1] == 'orders':
                datasets += [('products', products.query(_query)), ('promotions', promotions.query(_query))]

            _query = "process == 'connected'"
            if type[0] == 'dimension':
                _query += " and "

            for add in datasets:
                acts = []
                for a in add[1].to_dict('results'):
                    acts.append(create_data_access_parameters(a, index=index[1], date=None, test=False))
                data_config[index[1]][type[0]][add[0]] = acts
        return data_config


def get_data_connection_arguments(es_tag, data_config):
    conn = read_sql(
        """
        SELECT  * FROM data_connection  WHERE tag =  '""" + es_tag + "' and process = 'connected'", con)
    data_config = create_date_structure(conn, data_config)
    return data_config


def create_index(tag):
    """

    :return:
    """


def create_sample_data(tag):
    """

    :return:
    """


def connection_elasticsearch_check(request):
    """

    :param request: elasticsearch connected tag name
    :return:
    """

    url = request['url']
    if request['url'] == 'None':
        url = 'http://' + str(request['host']) + ':' + str(request['port'])

    try:
        res = requests.get(url)
        return True, 'Connected!'
    except Exception as e:
        print(e)
        return False, 'Connection Failed!', elasticsearch_connection_refused_comment


def connection_check(request, index='orders', type=''):
    """

    :param tag: elasticsearch connected tag name
    :return:
    """
    accept, message, data, raw_columns = False, "Connection Failed!", None, []
    try:
        args = create_data_access_parameters(request, index=index, date=None, test=5)
        print(args)
        gd = GetData(**args)
        gd.query_data_source()
        if gd.data is not None:
            if len(gd.data) != 0:
                _columns = list(gd.data.columns)
                _df = gd.data
                if len(_columns) >= acception_column_count[type + index]:  # required list; order_id, client, s_start_date, amount, has_purchased
                    # _df = check_data_integration(data=_df, index=index)
                    accept, message, data, raw_columns = True, 'Connected!', _df.to_dict('results'), gd.data.columns.values
    except Exception as e:
        print(e)
    return accept, message, data, raw_columns


def check_data_integration(data, index):
    columns = list(data.columns)
    for col in sample_data_columns[index + '_sample_data'][1:]:
        if col not in columns:
            data[col] = '....'
    data = data[sample_data_columns[index + '_sample_data'][1:]]

    return data

def initialize_elastic_search():
    """

    :return:
    """
