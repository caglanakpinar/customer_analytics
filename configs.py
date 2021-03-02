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

default_message = {'orders': '....',
                   'orders_data': '....',
                   'orders_columns': '....',
                   'downloads': '....',
                   'downloads_data': '....',
                   'downloads_columns': '....',
                   'action_orders': '....',
                   'action_downloads': '....',
                   'product_orders': '....',
                   'schedule': '...',
                   'schedule_columns': '...',
                   'schedule_tags': '....',
                   'logs': '....',
                   'last_log': '....'
                   }

acception_column_count = {'orders': 5, 'downloads': 2, 'action_orders': 2, 'action_downloads': 2,
                          'product_orders': 4, 'promotion_orders': 2}

schedule_columns = ["ID", "ElasticSearch Tag Name", "Is Dimension Connection", "Process",
                    "Sessions Connection Tag Name", "Sessions Connection Data Source", "Sessions Connection Data Query/Path",
                    "Customers Connection Tag Name", "Customers Connection Data Source", "Customers Connection Data Query/Path",
                    "Is Connection Data Source Of An Action of Sessions/Customers",
                    "Is Connection Data Source Of Products of Sessions",
                    "Is Connection Data Source Of Promotions of Sessions",
                    "Last Time Triggered Scheduling Job", "Schedule Time Period", "Schedule Status", "Has Columns been assigned yet"]