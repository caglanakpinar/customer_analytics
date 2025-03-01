import pandas as pd
from flask_login import current_user
import logging

from customeranalytics.data_storage_configurations import DataStorageConfigurations
from customeranalytics.app.home.forms import Charts
from customeranalytics.app.home.profiles import Profiles
from customeranalytics.app.home.search import Search
from customeranalytics.exploratory_analysis import ea_configs
from customeranalytics.ml_process import ml_configs


class RouterRequest(DataStorageConfigurations, Charts, Profiles, Search):
    def __init__(self):
        super().__init__()
        self.return_values = {}
        self.active_connections = False
        self.hold_connection = False
        self.recent_connection = False
        self.message = self.default_message
        self.success_data_execute = """ 
            Data Storage Process is initialized!
            This process mainly involves fetching data
            from the data sources and storing them into the ElasticSearch indexes.
            This will take a while. Data Storage Process is triggered for
        """

    def info_logs_for_chat(self, info):
        return {
        'user': 'info',
        'date': str(self.current_date_to_day())[0:19],
        'user_logo': 'info.jpeg',
        'chat_type': 'info', 'chart': '',
        'general_message': info,
        'message': ''
    }

    def logs_update(self, logs):
        """
        logs table in sqlite table is updated.
        chats table in sqlite table is updated.
        """
        self.check_for_table_exits(table='logs')
        self.check_for_table_exits(table='chat')

        try:
            logs['login_user'] = current_user
            logs['log_time'] = str(self.current_date_to_day())[0:19]
            self.execute_query(
                self.insert_query(
                    table='logs',
                    columns=self.sqlite_queries['columns']['logs'][1:],
                    values=logs
                )
            )
        except Exception as e: logging.error(e)

        try: self.execute_query(
            self.insert_query(
                table='chat',
                columns=self.sqlite_queries['columns']['chat'][1:],
                values=self.info_logs_for_chat(logs['info'])
            )
        )
        except Exception as e: print(e)

    def assign_color_for_es_tag(self, data):
        _unique_es_tags = list(data['tag'].unique())
        data = pd.merge(
            data,
            (
                pd.DataFrame(zip(_unique_es_tags, self.colors[0:len(_unique_es_tags)]))
                .rename(columns={0: "tag", 1: "color"})
            ),
            on='tag',
            how='left'
        )
        return data

    def get_intersect_columns_with_request(self, requests, table):
        return list(set(list(requests.keys())) & set(self.sqlite_queries['columns'][table][1:]))

    def check_for_data_source_connection(self, requests, columns):
        try:
            return self.connection_check(request={col: requests[col] for col in columns},
                                    index=requests['data_type'],
                                    type=requests['data_type'])
        except Exception as e: logging.error(e)

    def check_for_both_sessions_and_customers_data_source(self, data_connection):
        if data_connection['orders_data_source_tag'] not in ['None', None] and \
                data_connection['downloads_data_source_tag'] not in ['None', None]:
            return True
        else: return False

    def check_for_product_data_source(self, data_connection):
        if data_connection['products_data_source_tag'] not in ['None', None]:
            return True
        else: return False

    def check_for_insert_columns(self, columns, requests, table):
        try:
            for col in self.sqlite_queries['columns'][table][1:]:
                if col not in columns:
                    requests[col] = None
        except Exception as e: logging.error(e)
        return requests

    def check_for_session_and_customer_product_connect(self):
        data_connection = self.collect_data_from_table(table='data_connection')
        sessions, customers, products, deliveries = False, False, False, False
        if len(data_connection) != 0:
            data_connection = data_connection.to_dict('results')[-1]
            sessions = True if data_connection['orders_data_source_tag'] != 'None' else False
            customers = True if data_connection['downloads_data_source_tag'] != 'None' else False
            products = True if data_connection['products_data_source_tag'] != 'None' else False
            deliveries = True if data_connection['deliveries_data_source_tag'] != 'None' else False
        return {'sessions': str(sessions), 'customers': str(customers),
                'products': str(products), 'deliveries': str(deliveries)}

    def values_for_manage_data(self, template):
        es_connection = self.collect_data_from_table(table='es_connection')
        if len(es_connection) != 0:
            if template in ['add-data-purchase_2', 'add-data-product_2']:
                self.message['es_connection'] = es_connection.to_dict('results')[-1]
            else:
                self.message['es_connection'] = es_connection.to_dict('results')
            # self.message['s_c_p_connection_check'] = "_".join(
            #     [str(i) for i in self.check_for_session_and_customer_product_connect()])
            self.message['s_c_p_connection_check'] = self.check_for_session_and_customer_product_connect()

    def values_for_schedule_data(self):
        try:
            data_connection = self.collect_data_from_table(table='data_connection')
        except Exception as e:
            logging.error(e)
        prev_schedule = self.collect_data_from_table(table='schedule_data')
        if len(prev_schedule) != 0:
            self.message['schedule_check'] = True
        actions = self.collect_data_from_table(table='actions')
        if len(actions) != 0:
            actions = actions.groupby("data_type").agg(
                {"action_name": lambda x: ", ".join(list(x))}).reset_index().fillna('....')
        try:
            es_connection = self.collect_data_from_table(table='es_connection')
            if len(es_connection) != 0:
                self.message['es_connection'] = es_connection.to_dict('results')[-1]
            else:
                self.message['es_connection'] = '....'
        except Exception as e: logging.error(e)

        try:
            logs = self.collect_data_from_table(table='logs')
            if len(logs) != 0:
                logs['color'] = logs['color'].apply(lambda x: 'color:' + x + ';')
                self.message['logs'] = logs.to_dict('results')[-min(len(logs), 20):]
            else:
                self.message['logs'] = '....'
        except Exception as e: logging.error(e)

        if len(data_connection) != 0:
            if self.check_for_both_sessions_and_customers_data_source(data_connection.to_dict('results')[-1]):
                self.message['connect_accept'] = True
                for dt in ['orders', 'downloads', 'products', 'deliveries']:
                    data_connection[dt + '_data_query_path'] = self.sqlite_string_converter(
                        list(data_connection[dt + '_data_query_path'])[0], back_to_normal=True)
                data_connection = pd.concat([data_connection, prev_schedule], axis=1)
                data_connection = pd.concat([data_connection,
                                             actions.query("data_type == 'orders'").drop('data_type', axis=1).rename(
                                                 columns={"action_name": "ses_actions"}).reset_index()], axis=1).fillna('....')
                data_connection = pd.concat([data_connection,
                                             actions.query("data_type == 'downloads'").drop('data_type', axis=1).rename(
                                                 columns={"action_name": "d_actions"}).reset_index()], axis=1).fillna('....')

                schedule = data_connection.to_dict('records')[-1]
                self.message['schedule'] = {i: '....' for i in list(schedule.keys()) + ['ses_actions', 'd_actions'] +
                                            self.sqlite_queries['columns']['schedule_data'][1:]}
                for i in schedule:
                    if schedule[i] not in [None, 'None']:
                        self.message['schedule'][i] = schedule[i]
                self.message['schedule'] = [self.message['schedule']]
                if self.check_for_product_data_source(data_connection.to_dict('results')[-1]):
                    self.message['has_product_data_source'] = True

    def update_data_query_path_for_insert(self, requests):
        _data_type = requests['data_type']
        requests[_data_type + '_data_query_path'] = self.sqlite_string_converter(requests[_data_type + '_data_query_path'])
        return requests

    def update_data_connection_table(self, requests, columns):
        data_connections = self.collect_data_from_table(table='data_connection')
        requests = self.update_data_query_path_for_insert(requests)
        if len(data_connections) == 0:
            for col in self.sqlite_queries['columns']['data_connection'][1:]:
                if col not in list(requests.keys()):
                    requests[col] = None
            try:
                self.execute_query(
                    self.insert_query(
                        table='data_connection',
                        columns=self.sqlite_queries['columns']['data_connection'][1:],
                        values=requests
                    )
                )
            except Exception as e: logging.error(e)
        else:
            data_connections = data_connections.to_dict('results')[-1]
            try:
                self.execute_query(
                    self.update_query(
                        table='data_connection',
                        condition=" id = " + str(data_connections['id']),
                        columns=columns, values=requests
                    )
                )
            except Exception as e: logging.error(e)

    def update_data_columns_match_table(self, requests, columns):
        try:
            self.check_for_table_exits(table='data_columns_integration')
            data_columns_integration = self.collect_data_from_table(
                table='data_columns_integration',
                query_str=" id == 1"
            )
            if len(data_columns_integration) == 0:
                requests = self.check_for_insert_columns(
                    columns,
                    requests,
                    'data_columns_integration'
                )
                try:
                    self.execute_query(
                        self.insert_query(
                            table='data_columns_integration',
                            columns=self.sqlite_queries['columns']['data_columns_integration'][1:],
                            values=requests
                        )
                    )
                except Exception as e: logging.error(e)
            else:
                try:
                    self.execute_query(
                        self.update_query(
                            table='data_columns_integration',
                            condition=" id = 1 ",
                            columns=columns, values=requests
                        )
                    )
                except Exception as e: logging.error(e)
        except Exception as e: logging.error(e)

    def remove_data_type_action(self, requests):
        prev_actions = self.collect_data_from_table(table='actions')
        if len(prev_actions) != 0:
            prev_actions_data_type = prev_actions[prev_actions['data_type'] == requests['data_type']]
            if len(prev_actions_data_type) != 0:
                for a in prev_actions_data_type.to_dict('results'):
                    self.execute_query(
                        self.delete_query(
                            table='actions',
                            condition=" id = " + str(a['id'])
                        )
                    )

    def update_actions_table(self, requests):
        if requests['data_type'] not in ['products', 'deliveries']:
            self.remove_data_type_action(requests)
            self.check_for_table_exits(table='actions')
            if requests.get('actions', None) is not None:
                actions = []
                if requests['actions'] != '':
                    for i in [i.replace(" ", "") for i in requests['actions'].split(",")]:
                        counter = 0
                        for c in i:
                            if c == ' ':
                                counter += 1
                            else:
                                break
                        _action = i[counter:]
                        actions.append(_action)
                        self.execute_query(
                            self.insert_query(
                                table='actions',
                                columns=self.sqlite_queries['columns']['actions'][1:],
                                values={
                                    "action_name": _action,
                                    "data_type": requests['data_type']
                                }
                            )
                        )
                    requests['actions'] = ",".join(actions)
        else:
            self.remove_data_type_action(requests)

    def update_schedule_table(self, requests):
        try:
            self.check_for_table_exits(table='schedule_data')
            prev_schedule = self.collect_data_from_table(
                table='schedule_data',
                return_list=True
            )
            if len(prev_schedule) != 0:
                self.logs_update(
                    logs={
                        "page": "data-execute",
                        "info": "Previous job " + " is removed.",
                        "color": "red"
                    }
                )
                self.execute_query(
                    self.delete_query(
                        table='schedule_data',
                        condition=" id = 1"
                    )
                )
            columns = self.get_intersect_columns_with_request(
                requests,
                'schedule_data'
            )
            requests = self.check_for_insert_columns(
                columns,
                requests,
                'schedule_data'
            )
            requests['max_date_of_order_data'] = str(self.current_date_to_day())[0:19]
            self.execute_query(
                self.insert_query(
                    table='schedule_data',
                    columns=self.sqlite_queries['columns']['schedule_data'][1:],
                    values=requests
                )
            )
            self.logs_update(logs={"page": "data-execute",
                                   "info": self.success_data_execute + " ".join(requests['time_period'].split("_")),
                                   "color": "green"})
        except Exception as e:
            logging.error(e)
        return requests['tag']

    def update_data_query_path_on_schedule(self, request):
        keys = list(request.keys())
        data_source_query_path = [
            i
            for i in ['orders', 'downloads', 'products', 'deliveries']
            if i + '_data_query_path' in keys
        ]
        self.execute_query(
            self.update_query(
                table='data_columns_integration',
                condition=" id = 1 ",
                columns=data_source_query_path,
                values={
                    data_source_query_path: request[data_source_query_path[0]]
                }
            )
        )

    def update_message_and_tables(self):
        self.message = self.default_message

    def manage_data_integration(self, requests):
        if requests.get('connect', None) is not None:
            self.check_for_table_exits(table='es_connection')
            requests['port'] = str(self.default_es_port) if requests['port'] is None else requests['port']
            requests['host'] = str(self.default_es_host) if requests['host'] is None else requests['host']
            status, self.message['es_connection_check'] = self.check_elasticsearch(port=requests['port'],
                                                                              host=requests['host'],
                                                                              directory=requests['directory'])
            if status:
                try:
                    self.execute_query(
                        self.insert_query(table='es_connection',
                                                  columns=self.sqlite_queries['columns']['es_connection'][1:],
                                                  values=requests
                        )
                    )
                except Exception as e:
                    logging.error(e)

        if requests.get('delete', None) is not None:
            try:
                self.execute_query("DROP table es_connection")
            except Exception as e:
                logging.error(e)

    def data_connections(self, requests):
        # for orders index choose ElasticSearch Connection from es_connection table (only status == 'on')
        if requests.get('connect', None) is not None:
            self.check_for_table_exits(table='data_connection')
            self.check_for_table_exits(table='data_columns_integration')
            _columns = self.get_intersect_columns_with_request(requests, 'data_connection')
            _columns_2 = self.get_intersect_columns_with_request(requests, 'data_columns_integration')
            conn_status, self.message['data_source_con_check'], data, data_columns = self.check_for_data_source_connection(requests, _columns)

            # connection update
            if conn_status:
                self.update_actions_table(requests)
                self.update_data_connection_table(requests, _columns)
                self.update_data_columns_match_table(requests, _columns_2)

    def data_execute(self, requests):
        if requests.get('schedule', None) is not None:
            es_tag = self.update_schedule_table(requests)
            self.create_index(tag=es_tag, ea_configs=ea_configs, ml_configs=ml_configs)
        if requests.get('edit', None) is not None:
            self.update_data_query_path_on_schedule(requests)
        if requests.get('delete', None) is not None:
            try:
                self.execute_query("DELETE FROM schedule_data")
            except Exception as e:
                logging.error(e)

    def check_for_request(self, _r):
        _r_updated = {}
        try:
            for i in _r:
                _r_updated[i] = _r[i]
                if _r_updated[i] == 'True':
                    _r_updated[i] = True
                if _r_updated[i] == 'None':
                    _r_updated[i] = None
                if i == 'connect':
                    if _r_updated[i] not in [True, 'True']:
                        _r_updated['data_type'] = _r[i]
                        _r_updated['connect'] = True

                if i == 'schedule':
                    _r_updated['es_tag'] = _r[i]
                    _r_updated['schedule'] = True

        except Exception as e:
            logging.error(e)
        return _r_updated

    def execute_request(self, req, template):
        if req != {}:
            if template == 'data-es':
                self.manage_data_integration(self.check_for_request(req))
            if template in ['add-data-purchase', 'add-data-product', 'add-data-delivery']:
                self.data_connections(self.check_for_request(req))
            if template == 'data-execute':
                self.data_execute(self.check_for_request(req))

    def fetch_results(self, template, requests):
        if requests == {}:
            self.update_message_and_tables()
        else:
            if 'delete' in list(requests.keys()):
                self.update_message_and_tables()

        if template in ['add-data-purchase', 'add-data-product', 'add-data-delivery']:
            self.values_for_manage_data(template)
        if template == 'data-es':
            try:
                es_connection = self.collect_data_from_table(table='es_connection')
                if len(es_connection) != 0:
                    self.message['es_connection'] = es_connection.to_dict('results')[-1]
                else:
                    self.message['es_connection'] = '....'
            except Exception as e:
                logging.error(e)
        if template == 'data-execute':
            self.values_for_schedule_data()



