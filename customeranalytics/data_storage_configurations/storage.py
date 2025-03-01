from customeranalytics.data_storage_configurations.base import BaseDataStorageConfiguration
from customeranalytics.data_storage_configurations import Scheduler, GetData, QueryES


class DataStorageConfigurations(BaseDataStorageConfiguration):

    @staticmethod
    def create_connection_columns(index='orders') -> list[str]:
        return [
            index + i
            for i in [
                '_data_source_tag',
                '_data_source_type',
                '_data_query_path', '_password', '_user', '_port', '_host', '_db'
            ]
        ]

    def create_data_access_parameters(
            self,
            connection,
            index='orders',
            date=None,
            test=False
    ) -> dict[str, str]:
        return {
            'data_source': connection[index + '_data_source_type'],
            'date': date,
            'data_query_path': self.sqlite_string_converter(
                connection[index + '_data_query_path'],
                 back_to_normal=True
            ),
            'test': test,
            'config': {
                'host': connection[index + '_host'],
                'port': connection[index + '_port'],
                'password': connection[index + '_password'],
                'user': connection[index + '_user'],
                'db': connection[index + '_db']
            }
        }

    def get_data_connection_arguments(self) -> tuple[dict, list[str], dict[str, dict[str, str]]]:
        conn = self.collect_data_from_table(table="data_connection", return_list=True)[-1]
        columns = self.collect_data_from_table(table="data_columns_integration", return_list=True)[-1]

        data_configs = {
            'orders': self.create_data_access_parameters(conn, index='orders', date=None, test=False),
            'downloads': self.create_data_access_parameters(conn, index='downloads', date=None, test=False),
            'products': self.create_data_access_parameters(conn, index='products', date=None, test=False),
            'deliveries': self.create_data_access_parameters(conn, index='deliveries', date=None, test=False)
        }
        return conn, columns, data_configs

    def get_action_name(self) -> dict[str, list]:
        actions = self.read_query("SELECT * FROM actions ")
        a_orders, a_downloads = actions.query("data_type == 'orders'"), actions.query("data_type == 'downloads'")
        check_actions = lambda a: list(a['action_name']) if len(a) != 0 else []
        return {
            'orders': check_actions(a_orders),
            'downloads': check_actions(a_downloads)
        }

    def get_ea_and_ml_config(
            self, ea_configs, ml_configs, has_product_conn, has_promotion_conn, has_delivery_connection
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
        es_tag_conn = self.read_query(" SELECT  * FROM es_connection ")
        port, host, directory = [
            list(es_tag_conn[i])[0]
            for i in ['port', 'host', 'directory']
        ]
        actions = self.get_action_name()

        configs = []
        for conf in [ea_configs, ml_configs]:
            for ea in conf:
                if ea not in ['date', 'time_period']:
                    conf[ea]['host'] = host
                    conf[ea]['port'] = port
                if ea == 'funnel':
                    conf[ea]['actions'] = actions['downloads']
                    conf[ea]['purchase_actions'] = actions['orders']
                if ea in ['abtest', 'clv_prediction', 'delivery_anomaly']:
                    conf[ea]['temporary_export_path'] = directory
                if not has_product_conn:
                    if ea in ['products', 'abtest']:
                        conf[ea]['has_product_connection'] = False
                if not has_promotion_conn:
                    if ea in ['abtest', 'promotions']:
                        conf[ea]['has_promotion_connection'] = False
                if not has_delivery_connection:
                    if ea == 'delivery_anomaly':
                        conf[ea]['has_delivery_connection'] = False

            configs += [conf]
        return configs + [actions]

    @staticmethod
    def decision_for_product_conn(data_configs):
        return True if data_configs['products']['data_source'] not in [None, 'None'] else False

    @staticmethod
    def decision_for_delivery_conn(data_configs):
        return True if data_configs['deliveries']['data_source'] not in [None, 'None'] else False

    @staticmethod
    def decision_for_promotion_conn(columns):
        return True if columns['promotion_id'] not in [None, 'None'] else False

    def create_index(self, tag, ea_configs, ml_configs):
        """

        :return:
        """
        columns, data_configs = self.get_data_connection_arguments()[1:]
        has_product_connection = self.decision_for_product_conn(data_configs)
        has_delivery_connection = self.decision_for_delivery_conn(data_configs)
        has_promotion_connection = self.decision_for_promotion_conn(columns)
        _ea_configs, _ml_configs, _actions = self.get_ea_and_ml_config(
            ea_configs, ml_configs,
            has_product_connection,
            has_promotion_connection, has_delivery_connection
        )
        s = Scheduler(es_tag=tag,
                      data_connection_structure=data_configs,
                      ea_connection_structure=_ea_configs,
                      ml_connection_structure=_ml_configs, data_columns=columns, actions=_actions)
        s.run_schedule_on_thread(function=s.execute_schedule)

    def get_columns_condition(self, request, _columns, index):
        desire_column_count = 0
        if index == 'orders':
            a_col_count, p_col_count, d_col_count = 0, 0, 0
            if request.get('actions', None) is not None:
                a_col_count = len(request['actions'].split(","))
            if request.get('dimension', None) is not None:
                d_col_count = 1
            if request.get('promotion', None) is not None:
                p_col_count = 1
            desire_column_count = p_col_count + a_col_count + d_col_count + self.acception_column_count['orders']

        if index == 'downloads':
            a_col_count = 0
            if request.get('actions', None) is not None:
                a_col_count = len(request['actions'].split(","))
            desire_column_count = a_col_count + self.acception_column_count['downloads']

        if index == 'products':
            desire_column_count = self.acception_column_count['products']

        if index == 'deliveries':
            desire_column_count = self.acception_column_count['deliveries']

        if len(_columns) >= desire_column_count:
            return True
        else:
            return False

    def connection_check(self, request, index='orders', type=''):
        """

        :param tag: elasticsearch connected tag name
        :return:
        """
        accept, message, data, raw_columns = False, "Connection Failed", None, []
        try:
            args = self.create_data_access_parameters(
                request,
                index=index,
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
                            index
                    ):
                        accept, message, data, raw_columns = True, 'Connected', _df.to_dict(
                            'results'), gd.data.columns.values
        except Exception as e:
            print(e)
        return accept, message, data, raw_columns

    def check_data_integration(self, data, index):
        columns = list(data.columns)
        for col in self.sample_data_columns[index + '_sample_data'][1:]:
            if col not in columns:
                data[col] = '....'
        data = data[self.sample_data_columns[index + '_sample_data'][1:]]

        return data

    def check_elasticsearch(self, port, host, directory):
        """

        :return:
        """
        message = 'connected'
        connection = True
        if self.exists(directory):
            es = QueryES(port=port, host=host)
            if not es.es.ping():
                message = 'pls check the ElasticSearch connection.'
                connection = False

        else:
            message = 'pls check the directory.'
            connection = False
        return connection, message