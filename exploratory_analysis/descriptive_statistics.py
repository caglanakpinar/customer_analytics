import numpy as np
import pandas as pd
import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from configs import default_es_port, default_es_host, default_query_date, time_periods
from utils import *
from data_storage_configurations.query_es import QueryES


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
        -   daily_orders
        -   weekly_orders
        -   monthly_orders

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
                                  "payment_amount", "discount_amount", "actions.purchased"]
        self.stats = ["total_orders", "last_week_orders",
                      "total_revenue", "last_week_revenue",
                      "total_visitors", "last_week_visitors",
                      "total_discount", "last_week_discount",
                      "average_basket_value_per_user",
                      "daily_orders", "weekly_orders", "monthly_orders"]
        self.last_week = None
        self.time_periods = ["daily", "weekly", 'monthly']
        self.orders = pd.DataFrame()
        self.results = {}

    def get_data(self, start_date=None):
        """
        query orders index to collect the data with columns that are
        "id", "session_start_date", "client", "payment_amount", "discount_amount", "actions.purchased".
        :param start_date: starting date of query
        :return: data-farme individual order transactions.
        """
        start_date = default_query_date if start_date is None else start_date
        if len(self.orders) == 0:
            self.query_es = QueryES(port=self.port, host=self.host)
            self.query_es.query_builder(fields=self.orders_field_data,
                                        date_queries=[{"range": {"session_start_date": {"gte": start_date}}}])
            self.orders = pd.DataFrame(self.query_es.get_data_from_es(index=self.order_index))
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
                     [convert_dt_to_day_str, find_week_of_monday, convert_dt_to_month_str])):
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

    def daily_orders(self):
        """
        total number of orders per day
        :return: data-frame
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby("daily").agg(
            {"id": "count"}).reset_index().rename(columns={"id": "order_count"})

    def weekly_orders(self):
        """
        total number of orders per week
        :return: data-frame
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby("weekly").agg(
            {"id": "count"}).reset_index().rename(columns={"id": "order_count"})

    def monthly_orders(self):
        """
        total number of orders per month
        :return:
        """
        return self.orders[(self.orders['actions.purchased'] == True)].groupby("monthly").agg(
            {"id": "count"}).reset_index().rename(columns={"id": "order_count"})

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
                                            self.daily_orders,
                                            self.weekly_orders,
                                            self.monthly_orders])):
            print("stat name :", metric[0])
            if metric[0] in ["weekly_orders", "monthly_orders", "daily_orders"]:
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
            	weekly	            order_count
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