from os.path import abspath, join


elasticsearch_settings = {
                                    "settings": {
                                        "number_of_shards": 2,
                                        "number_of_replicas": 0,
                                        "index.mapping.total_fields.limit": 20000,
                                        "index.query.default_field": 10000000,
                                        "index.max_result_window": 10000000,
                                        "index": {
                                            "analysis": {

                                                "filter": {
                                                    "autocomplete_filter": {
                                                        "type": "edge_ngram",
                                                        "min_gram": 1,
                                                        "max_gram": 10
                                                    }
                                                },
                                                "analyzer": {
                                                    "autocomplete": {
                                                        "type": "custom",
                                                        "tokenizer": "standard",
                                                        "filter": [
                                                            "lowercase",
                                                            "autocomplete_filter"
                                                        ]
                                                    }
                                                }


                                            }
                                        }

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
                                            "analysis": {

                                                "filter": {
                                                    "autocomplete_filter": {
                                                        "type": "edge_ngram",
                                                        "min_gram": 1,
                                                        "max_gram": 10
                                                    }
                                                },
                                                "analyzer": {
                                                    "autocomplete": {
                                                        "type": "custom",
                                                        "tokenizer": "standard",
                                                        "filter": [
                                                            "lowercase",
                                                            "autocomplete_filter"
                                                        ]
                                                    }
                                                }



                                            },
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
                                            "analysis": {

                                                "filter": {
                                                    "autocomplete_filter": {
                                                        "type": "edge_ngram",
                                                        "min_gram": 1,
                                                        "max_gram": 10
                                                    }
                                                },
                                                "analyzer": {
                                                    "autocomplete": {
                                                        "type": "custom",
                                                        "tokenizer": "standard",
                                                        "filter": [
                                                            "lowercase",
                                                            "autocomplete_filter"
                                                        ]
                                                    }
                                                }





                                            },
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
default_es_bulk_insert_chunk_size = 10000
default_es_bulk_insert_chunk_bytes = 10485760000
max_elasticsearch_bulk_insert_bytes = 100000000
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
                        "payment_amount", "discount_amount", "basket", "total_products",
                        "session_start_date", "dimension"]

not_required_columns = {"orders": ['discount_amount'], 'downloads': ['signup_date'], 'products': ['category']}
not_required_default_values = {'discount_amount': float(0.0), 'signup_date': default_query_date, 'category': 'cat_1'}

downloads_index_columns = ["id", "download_date", "signup_date", "client"]

downloads_index_obj = {'id': 89481673,
                       'download_date': '2021-01-01T21:23:15',
                       'signup_date': '2021-01-14T09:22:15',
                       'client': 'u_100006'}

descriptive_stats = ["weekly_average_session_per_user", "weekly_average_order_per_user",
                     "purchase_amount_distribution", "weekly_average_payment_amount"]

descriptive_reports = ['monthly_orders',
                       'purchase_amount_distribution',
                       'weekly_average_order_per_user',
                       'weekly_average_session_per_user',
                       'user_counts_per_order_seq',
                       '',
                       'weekly_orders',
                       'weekly_average_payment_amount',
                       'daily_orders',
                       'hourly_orders']

abtest_promotions = ["order_and_payment_amount_differences",
                     "promotion_comparison",
                     "promotion_usage_before_after_amount_accept",
                     "promotion_usage_before_after_amount_reject",
                     "promotion_usage_before_after_orders_accept",
                     "promotion_usage_before_after_orders_reject"]

abtest_products = ["product_usage_before_after_amount_accept",
                   "product_usage_before_after_amount_reject",
                   "product_usage_before_after_orders_accept",
                   "product_usage_before_after_orders_reject"]

abtest_segments = ['segments_change_daily_before_after_amount',
                   'segments_change_daily_before_after_orders',
                   'segments_change_monthly_before_after_amount',
                   'segments_change_monthly_before_after_orders',
                   'segments_change_weekly_before_after_amount',
                   'segments_change_weekly_before_after_orders']


abtest_reports = ['product_usage_before_after_amount',
                  'product_usage_before_after_orders', 'promotion_comparison',
                  'promotion_usage_before_after_amount',
                  'promotion_usage_before_after_orders',
                  'segments_change_daily_before_after_amount',
                  'segments_change_daily_before_after_orders',
                  'segments_change_monthly_before_after_amount',
                  'segments_change_monthly_before_after_orders',
                  'segments_change_weekly_before_after_amount',
                  'segments_change_weekly_before_after_orders']

product_analytics = ['daily_products_of_sales',
                     'product_kpis',
                     'most_ordered_products',
                     'hourly_products_of_sales',
                     'most_combined_products',
                     'most_ordered_categories',
                     'hourly_categories_of_sales']

promotion_analytics = ['daily_organic_orders', 'daily_inorganic_ratio',
                       'daily_promotion_revenue', 'daily_promotion_discount',
                       'avg_order_count_per_promo_per_cust',
                       'promotion_number_of_customer', 'promotion_kpis',
                       'inorganic_orders_per_promotion_per_day', 'hourly_inorganic_ratio',
                       'hourly_organic_orders']


non_dimensional_reports = ["clv_prediction", "segmentation"]


clv_prediction_reports = ["daily_clv", "clvsegments_amount"]

chart_names = {
    "Sessions Of Actions Funnel":
        {"Daily Funnel": "funnel*daily_funnel",
         "Weekly Funnel": "funnel*weekly_funnel",
         "Monthly Funnel": "funnel*monthly_funnel",
         "Hourly Funnel": "funnel*hourly_funnel"},

    "Customers Of Actions Funnel":
        {"Daily Funnel": "funnel*daily_funnel_downloads",
         "Weekly Funnel": "funnel*weekly_funnel_downloads",
         "Monthly Funnel": "funnel*monthly_funnel_downloads",
         "Hourly Funnel": "funnel*hourly_funnel_downloads"},
    "Cohorts":
        {"Daily Download to 1st Order Cohort": "cohort*daily_cohort_downloads",
         "Daily From 1st to 2nd Order Cohort": "cohort*daily_cohort_from_1_to_2",
         "Daily From 2nd to 3rd Order Cohort": "cohort*daily_cohort_from_2_to_3",
         "Daily From 3rd to 4th Order Cohort": "cohort*daily_cohort_from_3_to_4",
         "Weekly Download to 1st Order Cohort": "cohort*weekly_cohort_downloads",
         "Weekly From 1st to 2nd Order Cohort": "cohort*weekly_cohort_from_1_to_2",
         "Weekly From 2nd to 3rd Order Cohort": "cohort*weekly_cohort_from_2_to_3",
         "Weekly From 3rd to 4th Order Cohort": "cohort*weekly_cohort_from_3_to_4"},
    "Descriptive Stats":
        {"Daily Orders": "stats*daily_orders",
         "Hourly Orders": "stats*hourly_orders",
         "Weekly Orders": "stats*weekly_orders",
         "Monthly Orders": "stats*monthly_orders",
         "Weekly Average Session Count per Customer": "descriptive*weekly_average_session_per_user",
         "Weekly Average Purchase Count per Customer": "descriptive*weekly_average_payment_amount",
         "Payment Amount Distribution": "descriptive*purchase_amount_distribution",
         "Weekly Average Payment Amount": "descriptive*weekly_average_payment_amount"},
    "Product Analytics":
        {"Most Combined Products": "product_analytic*most_combined_products",
         "Most Order Products": "product_analytic*most_ordered_products",
         "Most Order Categories": "product_analytic*most_ordered_categories"},
    "A/B test Promotion":
        {"Promotion Comparison": "abtest-promotion*promotion_comparison",
         "Order And Payment Amount Difference for Before And After Promotion Usage": "abtest-promotion*order_and_payment_amount_differences",
         "A/B Test Promotion B. - A. Time Periods Cust.s' Avg. Purchase Payment Amount Test (Test Accepted!)": "abtest-promotion*promotion_usage_before_after_amount_accept",
         "A/B Test Promotion B. - A. Time Periods Cust.s' Avg. Purchase Payment Amount Test (Test Rejected!)": "abtest-promotion*promotion_usage_before_after_amount_reject",
         "A/B Test Promotion B. - A. Time Periods Cust.s' Tot. Purchase Count Test (Test Accepted!)": "abtest-promotion*promotion_usage_before_after_orders_accept",
         "A/B Test Promotion B. - A. Time Periods Cust.s' Tott Purchase Count Test (Test Rejected!)": "abtest-promotion*promotion_usage_before_after_orders_reject"},
    "A/B Test Product":
        {
            "A/B Test Product - B. - A. Time Periods Cust.s' Avg. Purchase Payment Amount Test (Test Accepted!)": "abtest-product*product_usage_before_after_amount_accept",
            "A/B Test Product - B. - A. Time Periods Cust.s' Avg. Purchase Payment Amount Test (Test Rejected!)": "abtest-product*product_usage_before_after_amount_reject",
            "A/B Test Product - B. - A. Time Periods Cust.s' Totg Purchase Count Test (Test Accepted!)": "abtest-product*product_usage_before_after_orders_accept",
            "A/B Test Product - B. - A. Time Periods Cust.s' Totg Purchase Count Test (Test Rejected!)": "abtest-product*product_usage_before_after_orders_reject"},
    "A/B Test Customer Segment Change":
        {
            "A/B Test Cust. Segment Change - A/B Test Product - Daily Customers' Total Order Count per Customer Segment": "abtest-segments*segments_change_daily_before_after_orders",
            "Weekly Customers' Total Order Count per Customer Segment": "abtest-segments*segments_change_weekly_before_after_orders",
            "Monthly Customers' Total Order Count per Customer Segment": "abtest-segments*segments_change_weekly_before_after_orders",
            "Daily Customers' Average Purchase Payment Amount per Customer Segment": "abtest-segments*segments_change_daily_before_after_amount",
            "Weekly Customers' Average Purchase Payment Amount per Customer Segment": "abtest-segments*segments_change_weekly_before_after_amount",
            "Monthly Customers' Average Purchase Payment Amount per Customer Segment": "abtest-segments*segments_change_weekly_before_after_amount"},
    "RFM":
        {"RFM": "rfm*rfm",
         "Frequency - Recency": "rfm*frequency_recency",
         "Monetary - Frequency": "rfm*monetary_frequency",
         "Recency - Monetary": "rfm*recency_monetary"},
    "Segmentation":
        {"Customer Segmentation": "customer-segmentation*",
         "Frequency Segmentation": "customer-segmentation*frequency_clusters",
         "Monetary Segmentation": "customer-segmentation*monetary_clusters",
         "Recency Segmentation": "customer-segmentation*recency_clusters"},
    "CLV Prediction":
        {"Next Week CLV Predictions": "clv*daily_clv",
         "CLV Predicted Nex Week Customers of Segments of Total Purchase Amounts": "clv*daily_clv"},
    "Overall":
        {"Customer Journey": "index*customer_journey",
         "Total Number Customer Breakdown with Purchased Order Count": "clv*clvsegments_amount"},
    "Anomaly":
        {"Daily Funnel Anomaly": "anomaly*dfunnel_anomaly",
         "Daily Cohort Anomaly": "anomaly*dcohort_anomaly",
         "Daily Cohort Anomaly With Scores": "anomaly*dcohort_anomaly_2",
         "Daily Cohort Anomaly ": "anomaly*dorders_anomaly", "CLV RFM Vs Current RFM": "anomaly*clvrfm_anomaly"}
}


ALLOWED_IMAGE_EXTENSIONS = ["JPEG", "JPG", "PNG", "GIF"]
MAX_IMAGE_FILESIZE = 0.5 * 1024 * 1024

TIME_DIFF_STR = [(0, "1m ago"), (60, "1m ago"), (60 * 60, "m ago "), (60 * 60 * 24, "hr ago "),
                 (60 * 60 * 24 * 7, "d ago "), (60 * 60 * 24 * 7 * 4, "w ago "), (60 * 60 * 24 * 7 * 4 * 12, "y ago ")]

DATE_DIFF_STR = [(0, "today "), (60 * 60 * 24, "today"), (60 * 60 * 24 * 2, "yesterday")]

DATA_WORKS_READABLE_FORM = {'clv_prediction': 'CLV Prediction',
                            'abtest': 'A/B Test', 'funnel': 'Session Actions & Customers Actions Funnels',
                            'cohort': 'Cohorts', 'rfm': 'RFM', 'stats': 'Descriptive Statistics',
                            'products': 'Product Analytics', 'segmentation': 'Customer Segmentation',
                            'anomaly': 'Anomaly Detection', 'promotions': 'Promotion Analytics', 'churn': 'Churn'
                            }

none_types = [None, 'None', '-']
session_columns = {'order_id', 'client', 'session_start_date', 'date', 'payment_amount',
                   'discount_amount', 'has_purchased'}
customer_columns = {'client_2', 'download_date', 'signup_date'}
product_columns = {'order_id', 'product', 'price', 'category'}


data_types_for_search = {"product": [('chart_1', ['product_kpis']),
                                      ('chart_2', ['daily_products_of_sales']),
                                      ('chart_3', ['product_usage_before_after_amount_accept',
                                                   'product_usage_before_after_amount_reject']),
                                      ('chart_4', ['product_usage_before_after_orders_accept',
                                                   'product_usage_before_after_orders_reject']),
                                      ],
                         "promotion": [('chart_1', ['promotion_kpis']),
                                       ('chart_2', ['daily_inorganic_ratio']),
                                       ('chart_3', ['daily_promotion_revenue']),
                                       ('chart_4', ['daily_promotion_discount']),
                                       ],
                         "client": [('chart_1', ['client_kpis']),
                                    ('chart_2', ['client_feature_predicted'])],

                         "dimension": [('chart_1', ['dimension_kpis']),
                                       ('chart_2', ['daily_dimension_values']),
                                       ('chart_3', ['daily_dimension_values']),
                                       ('chart_4', ['daily_dimension_values'])],


                         }
