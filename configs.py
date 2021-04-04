from os.path import abspath, join


elasticsearch_settings = {
                                    "settings": {
                                        "number_of_shards": 2,
                                        "number_of_replicas": 0,
                                        "index.mapping.total_fields.limit": 20000,
                                        "index.query.default_field": 10000000,
                                        "index.max_result_window": 10000000,
                                        "index": {
                                            "analysis": {},
                                        },

                              },
                                "mappings": {
                                            "properties": {"date": {"type": "date"},
                                                           "session_start_date": {"type": "date"},
                                                           "discount_amount": {"type": "float"},
                                                           "payment_amount": {"type": "float"},
                                                           "actions": {
                                                               "properties": {
                                                                              "has_sessions": {"type": "boolean"},
                                                                              "has_basket": {"type": "boolean"},
                                                                              "order_screen": {"type": "boolean"},
                                                                              "purchased": {"type": "boolean"}
                                                                        }
                                                           }

                                                          }

                                        }

                            }

elasticsearch_settings_downloads = {
                                    "settings": {
                                        "number_of_shards": 2,
                                        "number_of_replicas": 0,
                                        "index.mapping.total_fields.limit": 20000,
                                        "index.query.default_field": 10000000,
                                        "index.max_result_window": 10000000,
                                        "index": {
                                            "analysis": {},
                                        },

                              },
                                "mappings": {
                                            "properties": {"download_date": {"type": "date"},
                                                           "signup_date": {"type": "date"}
                                                           }
                                        }
                            }


elasticsearch_settings_reports = {
                                    "settings": {
                                        "number_of_shards": 2,
                                        "number_of_replicas": 0,
                                        "index.mapping.total_fields.limit": 20000,
                                        "index.query.default_field": 10000000,
                                        "index.max_result_window": 10000000,
                                        "index": {
                                            "analysis": {},
                                        },

                              },
                                "mappings": {
                                            "properties": {
                                                            "report_date": {"type": "date"},
                                                            "report_types._to": {"type": "integer"},
                                                            "report_types._from": {"type": "integer"},
                                                            "frequency_segment": {"type": "float"},
                                                            "monetary_segment": {"type": "float"},
                                                            "recency_segment": {"type": "float"},
                                                           }
                                        }
                            }


default_sample_data_previous_day = 60
default_dask_partitions = 3
default_es_port = 9200
default_es_host = 'localhost'
time_periods = ["hourly", "daily", "weekly", 'monthly']
default_query_date = "1900-01-01T00:00:00"
query_path = join(abspath(__file__).split("configs.py")[0], "docs")
default_dask_partitions = 4
elasticsearch_connection_refused_comment = """
            Please check your directory. File path must be ending with ...../bin/elasticsearch.
            '/bin/elasticsearch' path is a folder in downloaded ES folder.
            Another option, running elasticsearch on terminal and assign 'port' and 'host'.
            ElasticSearch default port is 9200. If you running on local computer pls assign default host is 'localhost.' 
            """


schedule_columns = {
                    "tag": "ElasticSearch Tag Name",
                    "dimension": "Has Data Source Dimension?",
                    "orders_data_source_tag": "Sessions Connection Tag Name",
                    "orders_data_source_type": "Sessions Connection Data Source",
                    "orders_data_query_path": "Sessions Connection Data Query/Path",

                    "downloads_data_source_tag": "Customers Connection Tag Name",
                    "downloads_data_source_type": "Customers Connection Data Source",
                    "downloads_data_query_path": "Customers Connection Data Query/Path",

                    "products_data_source_tag": "Baskets Connection Tag Name",
                    "products_data_source_type": "Baskets Connection Data Source",
                    "products_data_query_path": "Baskets Connection Data Query/Path",

                    "s_action": "Has Sessions Data Source Actions?",
                    "d_action": "Has Customers Data Source Actions?",
                    "promotion_id": "Has Sessions Data Source Promotions?",

                    "max_date_of_order_data": "Last Time Triggered Scheduling Job",
                    "time_period": "Schedule Time Period",
                    "schedule_tag": "Schedule Status",
}


default_message = {'es_connection': '....',
                   'orders': '....',
                   'orders_data': '....',
                   'orders_columns': '....',
                   'downloads': '....',
                   'downloads_data': '....',
                   'downloads_columns': '....',
                   'action_orders': '....',
                   'action_downloads': '....',
                   'product_orders': '....',
                   'schedule': '...',
                   'schedule_columns': list(schedule_columns.values()),
                   'schedule_tags': '....',
                   'logs': '....',
                   'last_log': '....',
                   'active_connections': '....',
                   'connect_accept': False,
                   'has_product_data_source': False,
                   'es_connection_check': '....',
                   'schedule_check': False,
                   's_c_p_connection_check': 'False_False_False',
                   'data_source_con_check': '....'
                   }


acception_column_count = {'orders': 5, 'downloads': 2, 'products': 4}


orders_index_obj = {'id': 74915741,
   'date': None,
   'actions': {},
   'client': None,
   'promotion_id': None,
   'payment_amount': 0,
   'discount_amount': 0,
   'basket': {'p_10': {'price': 6.12, 'category': 'p_c_8', 'rank': 109},
    'p_145': {'price': 12.0, 'category': 'p_c_9', 'rank': 175},
    'p_168': {'price': 13.12, 'category': 'p_c_10', 'rank': 82},
    'p_9': {'price': 0.52, 'category': 'p_c_3', 'rank': 9},
    'p_4': {'price': 3.72, 'category': 'p_c_8', 'rank': 69},
    'p_104': {'price': 8.88, 'category': 'p_c_10', 'rank': 97},
    'p_74': {'price': 8.395, 'category': 'p_c_10', 'rank': 35}},
   'total_products': 7,
   'session_start_date': '2020-12-16T09:39:11'}
orders_index_columns = ["id", "date", "actions", "client", "promotion_id",
                        "payment_amount", "discount_amount", "basket", "total_products", "session_start_date", "dimension"]

not_required_columns = {"orders": ['discount_amount'], 'downloads': ['signup_date'], 'products': ['category']}
not_required_default_values = {'discount_amount': float(0.0), 'signup_date': default_query_date, 'category': 'cat_1'}

downloads_index_columns = ["id", "download_date", "signup_date", "client"]

downloads_index_obj = {'id': 89481673,
                       'download_date': '2021-01-01T21:23:15',
                       'signup_date': '2021-01-14T09:22:15',
                       'client': 'u_100006'}

