import pandas as pd

from customeranalytics.data_storage_configurations.base import BaseDataStorageConfiguration
from customeranalytics.data_storage_configurations.data_access import GetData


class DataStorageConfigurations(BaseDataStorageConfiguration):

    def __init__(self):
        super().__init__()

    def create_data_access_parameters(
            self,
            connection,
            data_type='orders',
            date=None,
            test=False
    ) -> dict[str, str] | None:
        if connection[data_type + '_data_source_type'] is not None:
            return {
                'data_source': connection[data_type + '_data_source_type'],
                'date': date,
                'data_query_path': self.sqlite_string_converter(
                    connection[data_type + '_data_query_path'],
                     back_to_normal=True
                ),
                'test': test,
                'config': {
                    'host': connection[f'{data_type}_host'],
                    'port': connection[f'{data_type}_port'],
                    'password': connection[f'{data_type}_password'],
                    'user': connection[f'{data_type}_user'],
                    'db': connection[f'{data_type}_db']
                }
            }
        return None

    def get_data_connection_arguments(self) -> tuple[dict, dict[str, dict[str, str]]]:
        conn = self.collect_data_from_table(table="data_connection")
        columns = self.collect_data_from_table(table="data_columns_integration")
        data_configs = {}
        for data_type in ['orders', 'downloads', 'products', 'deliveries']:
            _conn = self.create_data_access_parameters(conn, data_type=data_type)
            if _conn is not None:
                data_configs[data_type] = _conn

        return columns, data_configs

    def get_and_update_ea_and_ml_config(
            self
    ):
        """
            ea_configs = {"date": None,
                          "funnel": {"actions": ["download", "signup"],
                                     "purchase_actions": ["has_basket", "order_screen"],
                                     "host": 'localhost',
                                     "port": '9200',
                                     'download_index': 'downloads',
                                     'order_index': 'orders'},
                          "cohort": {"has_download": True, "host": 'localhost', "port": '9200',
                                     'download_index': 'downloads', 'order_index': 'orders'},
                          "product": {"has_product_connection": True, "has_download": True,
                                       "host": 'localhost', "port": '9200'},
                          "promotions": {"has_promotion_connection": True,
                                 "host": 'localhost', "port": '9200',
                                 "download_index": 'downloads', "order_index": 'orders'},
                          "rfm": {"host": 'localhost', "port": '9200',
                                  'download_index': 'downloads', 'order_index': 'orders'},
                          "stats": {"host": 'localhost', "port": '9200',
                                   'download_index': 'downloads', 'order_index': 'orders'}
                 }

            ml_configs = {"date": None,
                          'time_period': 'weekly',
                          "segmentation": {"host": 'localhost', "port": '9200',
                                           'download_index': 'downloads', 'order_index': 'orders'},
                          "clv_prediction": {"temporary_export_path": None,
                                             "host": 'localhost', "port": '9200',
                                             'download_index': 'downloads', 'order_index': 'orders', 'time_period': 'weekly'},
                          "abtest": {"has_product_connection": True, "temporary_export_path": None,
                                     "host": 'localhost', "port": '9200', 'download_index': 'downloads', 'order_index': 'orders'}
                         }

        """
        actions = {
            'orders': self.get_action_name('orders'),
            'downloads': self.get_action_name('downloads')
        }

        configs = []
        for config_type in ['ea_configs', 'ml_configs']:
            conf = getattr(self.connection_conf, config_type)
            for ea in conf:
                if ea == 'funnel':
                    conf[ea]['actions'] = actions['downloads']
                    conf[ea]['purchase_actions'] = actions['orders']
                if ea in ['abtest', 'clv_prediction', 'delivery_anomaly']:
                    conf[ea]['temporary_export_path'] = self.connection_folder
                if not self.decision_for_data_type_conn('products'):
                    if ea in ['products', 'abtest']:
                        conf[ea]['has_product_connection'] = False
                if self.connection_conf.data_connection.get('promotion_id') is not None:
                    if ea in ['abtest', 'promotions']:
                        conf[ea]['has_promotion_connection'] = False
                if not self.decision_for_data_type_conn('deliveries'):
                    if ea == 'delivery_anomaly':
                        conf[ea]['has_delivery_connection'] = False
            self.update_ea_ml_config(conf, config_type)
            configs += [conf]
        return configs + [actions]

    def decision_for_data_type_conn(self, data_type):
        return (
            True
            if self.connection_conf.data_connection.get(f'{data_type}_data_source_tag') is not None
            else False
        )

    def inject_data(self):
        columns, data_configs = self.get_data_connection_arguments()
        for data_type, args in data_configs.items():
            gd = GetData(**args)
            gd.query_data_source()
            _df = gd.data
            if not self.check_data_exists(data_type):
                self.insert_data(_df, data_type)
            else:
                self.update_data(_df, data_type)

    def data_works(self):
        """
        Execute Exploratory Analysis and Machine Learning Works which are implemented in the platform.
        This process is optional on the web interface so, it also checks 'is_mlworks' and 'is_exploratory'.
            Exploratory Analysis;
                - Funnels
                - Cohorts
                - Descriptive Statistics
                - RFM
                - Product Analytics
                - Promotion Analytics
            Machine Learning;
                - Customer Segmentation
                - CLV Prediction
                - A/B Test
                - Anomaly Detection

        These jobs are created per main and dimensional models individually but, are stored in the 'reports' index.
        """
        self.inject_data()

        _ea_configs, _ml_configs, _actions = self.get_and_update_ea_and_ml_config()
        args = dict(
            ml_connection_structure=_ml_configs,
            ea_connection_structure=_ea_configs,
            actions=_actions
        )
        self.data_work_pipelines_execution(
            **args
        )
        dimension_column = self.connection_conf.data_connection.get('dimension')
        if dimension_column is not None:
            for dim in self.get_data_from_folder()[dimension_column].unique():
                self.separator(dim=dim)
                args['dim'] = dim
                self.data_work_pipelines_execution(
                    **args
                )
        self.create_build_in_reports()

    def get_columns_condition(self, request, _columns, data_type):
        desire_column_count = 0
        if data_type == 'orders':
            a_col_count, p_col_count, d_col_count = 0, 0, 0
            if request.get('actions', None) is not None:
                a_col_count = len(request['actions'].split(","))
            if request.get('dimension', None) is not None:
                d_col_count = 1
            if request.get('promotion', None) is not None:
                p_col_count = 1
            desire_column_count = p_col_count + a_col_count + d_col_count + self.acception_column_count['orders']

        if data_type == 'downloads':
            a_col_count = 0
            if request.get('actions', None) is not None:
                a_col_count = len(request['actions'].split(","))
            desire_column_count = a_col_count + self.acception_column_count['downloads']

        if data_type == 'products':
            desire_column_count = self.acception_column_count['products']

        if data_type == 'deliveries':
            desire_column_count = self.acception_column_count['deliveries']

        if len(_columns) >= desire_column_count:
            return True
        else:
            return False

    def connection_check(self, request, data_type='orders'):
        """

        """
        accept, message, data, raw_columns = False, "Connection Failed", None, []
        try:
            args = self.create_data_access_parameters(
                request,
                data_type=data_type,
                date=None,
                test=5
            )
            gd = GetData(**args)
            gd.query_data_source()
            if gd.data is not None:
                if len(gd.data) != 0:
                    _columns = list(gd.data.columns)
                    _df = gd.data
                    # required list; order_id, client, s_start_date, amount, has_purchased
                    if self.get_columns_condition(
                            request,
                            _columns,
                            data_type
                    ):
                        accept, message, data, raw_columns = True, 'Connected', _df.to_dict(
                            'records'), gd.data.columns.values
        except Exception as e:
            print(e)
        return accept, message, data, raw_columns
