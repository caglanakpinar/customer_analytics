import logging

from customeranalytics import Config
from customeranalytics.app.home.profiles import Profiles
from customeranalytics.app.home.search import Search


class DefaultMessage:
    connection = Config.default_message_value
    orders = Config.default_message_value
    orders_data = Config.default_message_value
    orders_columns = Config.default_message_value
    downloads = Config.default_message_value
    downloads_data = Config.default_message_value
    downloads_columns = Config.default_message_value
    action_orders = Config.default_message_value
    action_downloads = Config.default_message_value
    product_orders = Config.default_message_value
    schedule = Config.default_message_value
    schedule_columns = Config.default_message_value
    schedule_tags = Config.default_message_value
    logs = Config.default_message_value
    last_log = Config.default_message_value
    active_connections = Config.default_message_value
    connect_accept = False
    has_product_data_source = False
    es_connection_check = Config.default_message_value
    schedule_check = False
    s_c_p_connection_check = 'False_False_False'
    data_source_con_check = Config.default_message_value


class RouterRequest(Profiles, Search):
    def __init__(self):
        super().__init__()
        self.return_values = {}
        self.active_connections = False
        self.hold_connection = False
        self.recent_connection = False
        self.message = DefaultMessage()
        self.success_data_execute = """ 
            Data Storage Process is initialized!
            This process mainly involves fetching data
            from the data sources and storing them into the Data Storage Folder.
            This will take a while. Data Storage Process is triggered for
        """

    def update_message(self):
        self.message = DefaultMessage()

    def info_logs_for_chat(self, info):
        return {
        'user': 'info',
        'date': str(self.current_date_to_day())[0:19],
        'user_logo': 'info.jpeg',
        'chat_type': 'info', 'chart': '',
        'general_message': info,
        'message': ''
    }

    def check_for_data_source_connection(self, requests):
        try:
            return self.connection_check(
                request=requests,
                data_type=requests['data_type'],
                type=requests['data_type']
            )
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

    def check_for_session_and_customer_product_connect(self) -> dict[str, str]:
        data_connection = self.collect_data_from_table(table='data_connection')
        sessions = data_connection.get('orders_data_source_tag', 'None')
        customers = data_connection.get('downloads_data_source_tag', 'None')
        products = data_connection.get('products_data_source_tag', 'None')
        deliveries = data_connection.get('deliveries_data_source_tag', 'None')

        return {
            'sessions': sessions,
            'customers': customers,
            'products': products,
            'deliveries': deliveries
        }

    def values_for_manage_data(self, template):
        if self.exists(self.connection_folder, self.connection_file_name):
            self.message.connection = 'True'
            self.message.s_c_p_connection_check = self.check_for_session_and_customer_product_connect()

    def values_for_schedule_data(self):
        data_connection = self.collect_data_from_table(table='data_connection')
        prev_schedule = self.collect_data_from_table(table='schedule_data')
        actions = self.collect_data_from_table(table='actions')
        logs = self.collect_data_from_table(table='logs')

        if prev_schedule.get('tag') is not None:
            self.message.schedule_check = True

        if logs[0].get('log_time') is not None:
            logs = logs[-min(len(logs), 20):]
            self.message.logs = []
            for l in logs:
                l['color'] = f"color:{l['color']};"
                self.message.logs.append(l)

        self.message.connect_accept = False
        self.message.has_product_data_source = False
        if (
            data_connection.get('orders_data_source_tag') is not None
            and data_connection.get('downloads_data_source_tag') is not None
        ):
            self.message.connect_accept = True
            for dt in ['orders', 'downloads', 'products', 'deliveries']:
                data_connection[dt + '_data_query_path'] = list(data_connection[dt + '_data_query_path'])[0]

            if data_connection.get('products_data_source_tag') is not None:
                self.message.has_product_data_source = True

            self.message.schedule = [{
                **data_connection,
                **prev_schedule,
                **actions
            }]

    def update_data_query_path_for_insert(self, requests):
        _data_type = requests['data_type']
        requests[_data_type + '_data_query_path'] = self.sqlite_string_converter(requests[_data_type + '_data_query_path'])
        return requests

    def update_actions_table(self, requests):
        if requests['data_type'] not in ['products', 'deliveries']:
            if requests.get('actions', '') != '':
                self.update_actions(
                    {
                        "action_name": [i.replace(" ", "") for i in requests['actions'].split(",")],
                        "data_type": requests['data_type']
                    }
                )

    def manage_data_integration(self, requests):
        if requests.get('connect', None) is not None:
            self.create_configuration(requests['directory'])

    def data_connections(self, requests):
        if requests.get('connect', None) is not None:
            (
                conn_status,
                self.message.data_source_con_check,
                data,
                data_columns
            ) = self.connection_check(
                request=requests,
                data_type=requests['data_type']
            )
            # connection update
            if conn_status:
                request = self.update_data_query_path_for_insert(requests)
                self.update_actions_table(requests)
                self.update_table('data_connection', request)
                self.update_table('data_columns_integration', request)


    def data_execute(self, requests):
        if requests.get('schedule', None) is not None:
            requests['max_date_of_order_data'] = str(self.current_date_to_day())[0:19]
            self.update_table('schedule_data', requests)
            self.run_schedule_on_thread(
                function=self.execute_schedule,
                args={
                    "jobs": self.data_works
                }
            )
            self.logs_update(
                logs={
                    "page": "data-execute",
                    "info": self.success_data_execute + " ".join(requests['time_period'].split("_")),
                    "color": "green"
                }
            )
        if requests.get('edit', None) is not None:
            self.update_table('data_columns_integration', requests)
        if requests.get('delete', None) is not None:
            self.remove_table('schedule_data')

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
                    _r_updated['schedule'] = True
        except Exception as e:
            logging.error(e)
        return _r_updated

    def execute_request(self, req, template):
        if req != {}:
            u_date_req = self.check_for_request(req)
            if template == 'data-conf':
                self.manage_data_integration(u_date_req)
            if template in ['add-data-purchase', 'add-data-product', 'add-data-delivery']:
                self.data_connections(u_date_req)
            if template == 'data-execute':
                self.data_execute(u_date_req)

    def fetch_results(self, template):
        if self.exists(self.connection_folder, self.connection_file_name):
            self.message.connection = 'True'
            self.message.s_c_p_connection_check = self.check_for_session_and_customer_product_connect()

        if template == 'data-execute':
            self.values_for_schedule_data()





