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
    if columns is not None:
        _ds['columns'] = columns
    return _ds


def main_connection_table_to_dictionary(data_config, main, index, columns):
    for m in main.to_dict('results'):
        try:
            _cols = columns[columns['tag'] == m[index[0]]].to_dict('results')[-1]
        except Exception as e:
            _cols = []
        if len(_cols) != 0:
            if 'True' not in [m[i] for i in ['is_action', 'is_product', 'is_promotion']]:
                data_config[index[1]]['main']['connection'] = create_data_access_parameters(m,
                                                                                            index=index[1], date=None,
                                                                                            test=False, columns=_cols)
            else:
                _type = None
                if m['is_action'] == 'True':
                    _type = 'action'
                if m['is_product'] == 'True' and index[1] == 'orders':
                    _type = 'product'
                if m['is_promotion'] == 'True' and index[1] == 'orders':
                    _type = 'promotion'
                if _type:
                    data_config[index[1]]['main'][_type].append(create_data_access_parameters(m,
                                                                                              index=index[1],
                                                                                              date=None,
                                                                                              test=False,
                                                                                              columns=_cols))
    return data_config


def dimension_connection_table_to_dictionary(data_config, main, index, columns):
    dimension_main = main.query("is_action == 'None' and is_product == 'None' and is_promotion == 'None'")
    additional_ds = {'action': main.query("is_action == 'True'"), 'product': main.query("is_action == 'True'"),
                     'promotion': main.query("is_promotion == 'True'")}
    _conn_default = data_config[index[1]]['dimensions'][0]
    data_config[index[1]]['dimensions'] = []
    for m in dimension_main.to_dict('results'):
        _conn = _conn_default
        _cols_dim = columns[columns['tag'] == m[index[0]]].to_dict('results')[-1]
        _conn['connection'] = create_data_access_parameters(m, index=index[1], date=None, test=False, columns=_cols_dim)
        # additional data sources
        for add in additional_ds:
            if index[1] == 'orders':
                _add_ds = additional_ds[add]
            else:
                _add_ds = additional_ds[add] if add == 'action' else []
            if len(_add_ds) != 0:
                _c = _add_ds[_add_ds['dimension'] == m['id']].to_dict('results')
                if len(_c) != 0:
                    _conn[add] = []
                    for _a in _c:
                        _cols_dim_add = columns[columns['tag'] == m[index[0]]].to_dict('results')[-1]
                        _conn[add].append(create_data_access_parameters(_a, index=index[1],
                                                                        date=None, test=False, columns=_cols_dim_add))
        data_config[index[1]]['dimensions'].append({m['orders_data_source_tag']: _conn})
    return data_config


def create_date_structure(es_tag_connections, columns, data_config):
    for index in [("orders_data_source_tag", "orders"), ("downloads_data_source_tag", "downloads")]:
        print()
        conns = es_tag_connections.query(index[0] + " == " + index[0])
        _columns = create_connection_columns(index=index[1])
        _query = " process == 'connected' "
        main = conns.query(_query + " and dimension == 'None' ")# [_columns]
        dimensions = conns.query(_query + " and dimension != 'None' ") # [_columns]

        # main connections
        try:
            data_config = main_connection_table_to_dictionary(data_config, main, index, columns)
        except Exception as e:
            print(e)
        # dimension connections
        if len(dimensions) != 0:
            try:
                data_config = dimension_connection_table_to_dictionary(data_config, dimensions, index, columns)
            except Exception as e:
                print(e)
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
    s = Scheduler(es_tag=tag, data_connection_structure=get_data_connection_arguments(tag, data_config))
    s.run_schedule_on_thread()

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
