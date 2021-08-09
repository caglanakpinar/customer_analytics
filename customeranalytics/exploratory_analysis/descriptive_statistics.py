import numpy as np
import pandas as pd
import sys, os, inspect
import warnings
warnings.filterwarnings("ignore")

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_es_port, default_es_host, default_query_date, time_periods
from customeranalytics.utils import *
from customeranalytics.data_storage_configurations.query_es import QueryES


class Stats:
    """
    There are some overall values that need to check each day for businesses.
    These values are also crucial metrics for the dashboards.

    Here are the basic aggregated values;
        -   total_orders
        -   last_week_orders
        -   total_revenue
        -   last_week_revenue
        -   total_visitors
        -   last_week_visitors
        -   total_discount
        -   last_week_discount
        -   average_basket_value_per_user
        -   overall_payment_distribution
        -   weekly_average_order_per_user
        -   weekly_average_session_per_user
        -   weekly_average_payment_amount
        -   daily_orders
        -   weekly_orders
        -   monthly_orders
        -   purchase_amount_distribution



    !!!!
    ******* ******** *****
    Dimensional Stats:
    Descriptive Statistics must be created individually for dimensions.
    For instance, the Data set contains locations dimension.
    In this case, each location of 'orders' and 'downloads' indexes must be created individually.
    by using 'download_index' and 'order_index' dimension can be assigned in order to create the descriptive Stats

    download_index; downloads_location1 this will be the location dimension of
                    parameters in order to query downloads indexes; 'location1'.
    download_index; orders_location1 this will be the location dimension of
                    parameters in order to query orders indexes; 'location1'.
    ******* ******** *****
        !!!
    """

    def __init__(self,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        ******* ******** *****
        Dimensional Stats:
        Descriptive Statistics must be created individually for dimensions.
        For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimension can be assigned in order to create the descriptive Stats

        download_index; downloads_location1 this will be the location dimension of
                        parameters in order to query downloads indexes; 'location1'.
        download_index; orders_location1 this will be the location dimension of
                        parameters in order to query orders indexes; 'location1'.
        ******* ******** *****
        !!!

        :param host: elasticsearch host
        :param port: elasticsearch port
        :param download_index: elasticsearch port
        :param order_index: elasticsearch port
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.download_index = download_index
        self.order_index = order_index
        self.query_es = QueryES(port=port, host=host)
        self.orders_field_data = ["id", "session_start_date", "client",
                                  "payment_amount", "discount_amount", "actions.purchased", "dimension"]
        self.stats = ["total_orders", "last_week_orders",
                      "total_revenue", "last_week_revenue",
                      "total_visitors", "last_week_visitors",
                      "total_discount", "last_week_discount",
                      "average_basket_value_per_user",
                      "hourly_orders", "daily_orders", "weekly_orders", "monthly_orders",
                      "purchase_amount_distribution", "weekly_average_order_per_user",
                      "weekly_average_session_per_user", "weekly_average_payment_amount", "user_counts_per_order_seq",
                      "hourly_revenue", "daily_revenue", "weekly_revenue", "monthly_revenue",
                      "total_order_count_per_customer", "dimension_kpis", "daily_dimension_values"]
        self.last_week = None
        self.time_periods = time_periods  # ["daily", "weekly", 'monthly']
        self.orders = pd.DataFrame()
        self.dimension_kpis = pd.DataFrame()
        self.daily_dimension_values = pd.DataFrame()
        self.results = {}

    def dimensional_query(self, boolean_query=None):
        if dimension_decision(self.order_index):
            if boolean_query is None:
                boolean_query = [{"term": {"dimension": self.order_index}}]
            else:
                boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def get_data(self, start_date=None):
        """
        query orders index to collect the data with columns that are
        "id", "session_start_date", "client", "payment_amount", "discount_amount", "actions.purchased".
        :param start_date: starting date of query
        :return: data-frame individual order transactions.
        """
        start_date = default_query_date if start_date is None else start_date
        if len(self.orders) == 0:
            self.query_es = QueryES(port=self.port, host=self.host)
            self.query_es.query_builder(fields=self.orders_field_data,
                                        date_queries=[{"range": {"session_start_date": {"gte": start_date}}}],
                                        boolean_queries=self.dimensional_query())
            self.orders = pd.DataFrame(self.query_es.get_data_from_es())
            self.orders['date'] = self.orders['session_start_date'].apply(lambda x: convert_to_date(x))

    def get_last_week(self):
        """
        in order to compare the recent week and last week the previous date of the whole order data is found.
        :return: datetime
        """
        self.last_week = max(self.orders['date']) - datetime.timedelta(days=7)

    def get_time_period(self):
        """
        converting date column of  values into the time_periods (hourly weekly, monthly,..)
        orders; total data (orders/downloads data with actions)
        final data; data set with time periods
        """
        for p in list(zip(self.time_periods,
                     [convert_str_to_hour, convert_dt_to_day_str, find_week_of_monday, convert_dt_to_month_str])):
            self.orders[p[0]] = self.orders["date"].apply(lambda x: p[1](x))

    def total_orders(self):
        """
        Total number of orders stored in orders index
        :return: integer
        """
        return len(self.orders[self.orders['actions.purchased'] == True]['id'])

    def last_week_orders(self):
        """
        Total number of orders in last 7 days
        :return: integer
        """
        return len(self.orders[(self.orders['actions.purchased'] == True) &
                               (self.orders['date'] > self.last_week)]['id'])

    def total_revenue(self):
        """
        Total payment amount of orders
        :return: float
        """
        return sum(self.orders[self.orders['actions.purchased'] == True]['payment_amount'])

    def last_week_revenue(self):
        """
        Total payment amount of orders in last 7 days
        :return: float
        """
        return sum(self.orders[(self.orders['actions.purchased'] == True) &
                               (self.orders['date'] > self.last_week)]['payment_amount'])

    def total_visitors(self):
        """
        Total number of unique visitors
        :return: integer
        """
        return len(self.orders['client'].unique())

    def last_week_visitors(self):
        """
        Total number of unique visitors in last 7 days
        :return: integer
        """
        return len(self.orders['client'].unique())

    def total_discount(self):
        """
        Total discount amount of orders
        :return: float
        """
        return sum(self.orders[self.orders['actions.purchased'] == True]['discount_amount'])

    def last_week_discount(self):
        """
        Total discount amount of orders in last 7 days
        :return: integer
        """
        return sum(self.orders[(self.orders['actions.purchased'] == True) &
                               (self.orders['date'] > self.last_week)]['discount_amount'])

    def average_basket_value_per_user(self):
        """
        Average payment amount of orders per user who has orders
        :return: float
        """
        return np.mean(self.orders[(self.orders['actions.purchased'] == True)].groupby("client").agg(
            {"payment_amount": "mean"}).reset_index()['payment_amount'])

    def hourly_orders(self):
        """
        Average of total order count per day
        :return: data-frame
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby('hourly').agg(
            {'id': 'count'}).reset_index().rename(columns={'id': 'orders'}).groupby('hourly').agg(
            {'orders': 'mean'}).reset_index()

    def daily_orders(self):
        """
        total number of orders per day
        :return: data-frame
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby("daily").agg(
            {"id": "count"}).reset_index().rename(columns={"id": "orders"})

    def weekly_orders(self):
        """
        total number of orders per week
        :return: data-frame
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby("weekly").agg(
            {"id": "count"}).reset_index().rename(columns={"id": "orders"})

    def monthly_orders(self):
        """
        total number of orders per month
        :return:
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby("monthly").agg(
            {"id": "count"}).reset_index().rename(columns={"id": "orders"})

    def hourly_revenue(self):
        """
        total revenue per hour
        :return: data-frame
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby('hourly').agg(
            {"payment_amount": "sum"}).reset_index()

    def daily_revenue(self):
        """
        total revenue per day
        :return: data-frame
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby("daily").agg(
            {"payment_amount": "sum"}).reset_index()

    def weekly_revenue(self):
        """
        total revenue per week
        :return: data-frame
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby("weekly").agg(
            {"payment_amount": "sum"}).reset_index()

    def monthly_revenue(self):
        """
        total revenue per month
        :return:
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby("monthly").agg(
            {"payment_amount": "sum"}).reset_index()

    def purchase_amount_distribution(self):
        """
        Payment values (purchased orders) of Distribution
        :return: data-frame; payment_bins, orders
        """
        _orders = self.orders[(self.orders['actions.purchased'] == True)]
        _amount = list(_orders['payment_amount'])
        _min_amount, _max_amount = min(_amount), max(_amount)
        _range = (_max_amount - _min_amount) / 20

        bins = {}
        for i in range(0, 20):
            bins[i] = {"min": _min_amount + (_range * i), "max": _min_amount + (_range * (i + 1))}

        _orders['payment_bins'] = _orders['payment_amount'].apply(
            lambda x: str(round(_min_amount + (int((x - _min_amount) / _range) * _range), 2)) if x == x else '-')
        _orders = _orders.groupby("payment_bins").agg({"id": "count"}).reset_index().rename(columns={"id": "orders"})
        _orders['payment_bins'] = _orders['payment_bins'].apply(lambda x: str(x))
        return _orders

    def weekly_average_order_per_user(self):
        """
        Weekly average order per user
        """
        _orders = self.orders[(self.orders['actions.purchased'] == True)].groupby(["weekly", "client"]).agg(
            {"id": "count"}).reset_index().rename(columns={"id": "orders"})
        return _orders.groupby("weekly").agg({"orders": "mean"}).reset_index()

    def weekly_average_session_per_user(self):
        """
        Weekly average session per user
        """
        _orders = self.orders.groupby(["weekly", "client"]).agg(
            {"id": "count"}).reset_index().rename(columns={"id": "sessions"})

        return _orders.groupby("weekly").agg({"sessions": "mean"}).reset_index()

    def weekly_average_payment_amount(self):
        """
        Weekly average payment amount
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby("weekly").agg(
            {"payment_amount": "mean"}).reset_index()

    def user_order_count_per_order_seq(self):
        """
        Customers' number of orders are calculated. The number of unique customer per total order count are calculated.
        """
        self.orders['order_seq_num'] = self.orders.sort_values(by=['client', 'date'],
                                                               ascending=True).groupby(['client'])['client'].cumcount() + 1
        self.orders_freq = self.orders.query("order_seq_num != 1")
        self.orders_freq = self.orders_freq.groupby("order_seq_num").agg({"id": "count"}).reset_index()
        self.orders_freq = self.orders_freq.sort_values(by='order_seq_num', ascending=True)
        self.orders_freq = self.orders_freq.rename(columns={"id": "frequency"})
        self.orders_freq['order_seq_num'] = self.orders_freq['order_seq_num'].apply(lambda x: str(x))
        return self.orders_freq

    def get_customer_total_order_count(self):
        """
        total order count per customer
        """
        return self.orders.groupby("client").agg(
            {"id": lambda x: len(np.unique(x))}).reset_index().rename(columns={"id": "order_count"})

    def get_dimension_kpis(self):
        if not dimension_decision(self.order_index):
            _dimensions = list(self.orders['dimension'].unique())
            print(_dimensions)
            if len(_dimensions) > 1:
                self.dimension_kpis = self.orders.groupby(["dimension"]).agg(
                    {"id": lambda x: len(np.unique(x)), "payment_amount": "sum",
                     "discount_amount": "sum", "client": lambda x: len(np.unique(x))
                     }).reset_index().rename(columns={"id": "order_count", "client": "client_client_count"})
        return self.dimension_kpis

    def get_daily_dimension_values(self):
        if not dimension_decision(self.order_index):
            _dimensions = list(self.orders['dimension'].unique())
            if len(_dimensions) > 1:
                self.daily_dimension_values = self.orders.groupby(["dimension", "daily"]).agg(
                    {"id": lambda x: len(np.unique(x)), "payment_amount": "sum",
                     "discount_amount": "sum", "client": lambda x: len(np.unique(x))
                     }).reset_index().rename(columns={"id": "order_count", "client": "client_count"})
        return self.daily_dimension_values

    def execute_descriptive_stats(self, start_date=None):
        """
        1.  Get order data from the order index.
        2.  Create time periods, weekly, daily, monthly
        3.  Find last week start date
        4.  Iterate metrics in order to calculate individually
        5.  Insert separate metrics which are data-frame stored individually.
            The rest of them are merged and stored as a one-row data-frame.
        """
        self.get_data(start_date=start_date)
        self.get_time_period()
        self.get_last_week()
        for metric in list(zip(self.stats, [self.total_orders, self.last_week_orders,
                                            self.total_revenue, self.last_week_revenue,
                                            self.total_visitors, self.last_week_visitors,
                                            self.total_discount, self.last_week_discount,
                                            self.average_basket_value_per_user,
                                            self.hourly_orders,
                                            self.daily_orders,
                                            self.weekly_orders,
                                            self.monthly_orders,
                                            self.purchase_amount_distribution,
                                            self.weekly_average_order_per_user, self.weekly_average_session_per_user,
                                            self.weekly_average_payment_amount,
                                            self.user_order_count_per_order_seq,
                                            self.hourly_revenue, self.daily_revenue,
                                            self.weekly_revenue, self.monthly_revenue,
                                            self.get_customer_total_order_count,
                                            self.get_dimension_kpis,
                                            self.get_daily_dimension_values
                                            ])):
            print("stat name :", metric[0])
            if metric[0] in ["hourly_revenue", "daily_revenue", "weekly_revenue", "monthly_revenue",
                             "hourly_orders", "weekly_orders", "monthly_orders", "daily_orders",
                             "purchase_amount_distribution", "weekly_average_order_per_user",
                             "weekly_average_session_per_user",
                             "weekly_average_payment_amount", "user_counts_per_order_seq",
                             "total_order_count_per_customer", "dimension_kpis", "daily_dimension_values"]:
                self.insert_into_reports_index(metric[1]().to_dict('results'),
                                               start_date,
                                               filters={"type": metric[0]},
                                               index=self.order_index)
            else:
                self.results[metric[0]] = metric[1]()
        self.insert_into_reports_index([self.results],
                                       start_date,
                                       filters={"type": ''},
                                       index=self.order_index)

    def insert_into_reports_index(self, stats, start_date, filters={}, index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": start_date or current date,
         "report_name": "stats",
         "index": "main",
         "report_types": {
                          "type": "overall", "weekly_orders", "daily_orders", "monthly_orders"
                          },
         "data": stats (list of dictionaries)
         }
        :param stats: overall, weekly_orders, daily_orders, monthly_orders
        :param start_date: datetime
        :param filters: {"type": "overall" or "weekly_orders" or "daily_orders" or "monthly_orders"}
        :param index: dimensionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = [{"id": np.random.randint(200000000),
                        "report_date": current_date_to_day().isoformat() if start_date is None else start_date,
                        "report_name": "stats",
                        "index": get_index_group(index),
                        "report_types": filters,
                        "data": stats}]

        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def fetch(self, stats, start_date=None):
        """
        query format;
            queries = {"stats": "overall"}
            queries = {"stats": "weekly_orders"}
            	weekly	            orders
            0	2020-12-07T00:00:00	3
            1	2020-12-14T00:00:00	36687
            2	2020-12-21T00:00:00	38166
        :param stats:  overall, weekly_orders, daily_orders, monthly_orders
        :param start_date:
        :return: data-frame
        """

        boolean_queries = [{"term": {"report_name": "stats"}},
                           {"term": {"report_types.type": stats}},
                           {"term": {"index": get_index_group(self.order_index)}}]
        date_queries = []
        if start_date is not None:
            date_queries = [{"range": {"report_date": {"gte": convert_to_iso_format(start_date)}}}]

        self.query_es = QueryES(port=self.port,
                                host=self.host)
        self.query_es.query_builder(fields=None, _source=True,
                                    boolean_queries=boolean_queries,
                                    date_queries=date_queries)
        _res = self.query_es.get_data_from_es(index="reports")
        _data = pd.DataFrame()
        if len(_res) != 0:
            _data = pd.DataFrame(_res[0]['_source']['data'])
        return _data