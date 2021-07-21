import numpy as np
import pandas as pd
import datetime
import random
from time import gmtime, strftime
import pytz
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import argparse
from itertools import product

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_es_port, default_es_host, none_types
from customeranalytics.utils import current_date_to_day, get_index_group, convert_str_to_hour, convert_to_date, convert_dt_to_day_str
from customeranalytics.utils import dimension_decision, convert_to_iso_format
from customeranalytics.data_storage_configurations.query_es import QueryES


class PromotionAnalytics:
    """

    """
    def __init__(self,
                 has_promotion_connection=True,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        !!!!
        ******* ******** *****
        Dimensional Promotion Analytic:
        Promotion must be created individually for dimensions. For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimension can be assigned in order to create a Promotion Analyse

        download_index; downloads_location1 this will be the location dimension of
                        parameters in order to query downloads indexes; 'location1'.
        download_index; orders_location1 this will be the location dimension of
                        parameters in order to query orders indexes; 'location1'.
        ******* ******** *****
        !!!

        :param has_promotion_connection: has_promotion_connection if there is no products data then, no promotion analytics
        :param host: elasticsearch host
        :param port: elasticsearch port
        :param download_index: elasticsearch port
        :param order_index: elasticsearch port
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.has_promotion_connection = has_promotion_connection
        self.download_index = download_index
        self.order_index = order_index
        self.query_es = QueryES(port=port, host=host)
        self.fields_promotions = ["id", "client", "promotion_id", "session_start_date",
                                  "payment_amount", "discount_amount"]
        self.promotions = pd.DataFrame()
        self.time_periods = ["hourly", "daily"]
        self.inorganic_orders = pd.DataFrame()
        self.inorganic_orders_hour = pd.DataFrame()
        self.daily_promotion_revenue = pd.DataFrame()
        self.daily_promotion_discount = pd.DataFrame()
        self.promotion_number_of_customer = pd.DataFrame()
        self.analysis = ["inorganic_orders_per_promotion_per_day", "daily_organic_orders", "hourly_organic_orders",
                         "daily_inorganic_ratio", "hourly_inorganic_ratio",
                         "daily_promotion_revenue", "daily_promotion_discount", "avg_order_count_per_promo_per_cust",
                         "promotion_number_of_customer", "promotion_kpis"]

    def dimensional_query(self, boolean_query=None):
        if dimension_decision(self.order_index):
            if boolean_query is None:
                boolean_query = [{"term": {"dimension": self.order_index}}]
            else:
                boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def get_time_period(self):
        """
        converting date column of  values into the time_periods (hourly weekly, monthly,..)
        orders; total data (orders/downloads data with actions)
        final data; data set with time periods
        """
        for p in list(zip(self.time_periods,
                     [convert_str_to_hour, convert_dt_to_day_str])):
            self.promotions[p[0]] = self.promotions["session_start_date"].apply(lambda x: p[1](x))

    def get_promotions(self, end_date):
        """
            1.  Fetch data from orders with filter; actions.purchased: True with promotion Id
        :param end_date: last date of data set
        """
        self.query_es = QueryES(port=self.port, host=self.host)
        self.query_es.date_queries_builder({"session_start_date": {"lt": end_date}})
        self.query_es.query_builder(fields=None, _source=True,
                                    boolean_queries=self.dimensional_query([{"term": {"actions.purchased": True}}]))
        self.promotions = self.query_es.get_data_from_es()
        self.promotions = pd.DataFrame(
            [{col: r['_source'][col] for col in self.fields_promotions} for r in self.promotions])

        self.promotions['payment_amount'] = self.promotions['payment_amount'].apply(lambda x: float(x))
        self.promotions['session_start_date'] = self.promotions['session_start_date'].apply(lambda x: convert_to_date(x))
        print(self.promotions.query("promotion_id != promotion_id"))
        self.promotions['has_promotion'] = self.promotions['promotion_id'].apply(lambda x: True if x not in none_types else False)
        self.get_time_period()
        print(self.promotions.head())

    def get_inorganic_orders_per_promotion_per_day(self):
        """
        Orders which have promotion columns with ID not null values.
        This process indicates total promoted order count per day per promotion_id.
        """
        self.inorganic_orders = self.promotions.query("has_promotion == True").groupby(["daily", "promotion_id"]).agg(
            {"id": lambda x: len(np.unique(x))}).reset_index().rename(columns={"id": "order_count"})
        return self.inorganic_orders

    def get_inorganic_orders_per_promotion_per_hour(self):
        """
        Orders which have promotion columns with ID not null values.
        This process indicates total promoted order count per hour per promotion_id.
        """
        self.inorganic_orders_hour = self.promotions.query("has_promotion == True").groupby(["hourly", "promotion_id"]).agg(
            {"id": lambda x: len(np.unique(x))}).reset_index().rename(columns={"id": "order_count"})
        return self.inorganic_orders_hour

    def get_organic_orders_per_day(self):
        """
        Orders which have promotion columns with null values.
        This process indicates total **NOT** promoted order count per hour per promotion_id.
        """
        return self.promotions.query("has_promotion != True").groupby("daily").agg(
            {"id": lambda x: len(np.unique(x))}).reset_index().rename(columns={"id": "order_count"})

    def get_daily_inorganic_ratio(self):
        """
        daily organic orders / daily total order count
        """
        daily_inorganic_ratio = self.promotions.groupby(["daily", "promotion_id"]).agg(
            {"id": lambda x: len(np.unique(x))}).reset_index().rename(columns={"id": "total_order_count"})
        daily_inorganic_ratio = pd.merge(daily_inorganic_ratio, self.inorganic_orders, how='left', on=["daily", "promotion_id"])
        daily_inorganic_ratio['inorganic_ratio'] = daily_inorganic_ratio['order_count'] / daily_inorganic_ratio['total_order_count']
        return daily_inorganic_ratio

    def get_hourly_inorganic_ratio(self):
        """
        hourly organic orders / hourly total order count
        """
        hourly_inorganic_ratio = self.promotions.groupby(["hourly", "promotion_id"]).agg(
            {"id": lambda x: len(np.unique(x))}).reset_index().rename(columns={"id": "total_order_count"})
        hourly_inorganic_ratio = pd.merge(hourly_inorganic_ratio, self.inorganic_orders_hour, how='left', on=["hourly", "promotion_id"])
        hourly_inorganic_ratio['inorganic_ratio'] = hourly_inorganic_ratio['order_count'] / hourly_inorganic_ratio['total_order_count']
        return hourly_inorganic_ratio

    def get_daily_promotion_revenue(self):
        """
        Orders which have promotion columns with null values.
        This process indicates total **NOT** promoted payment amount per hour per promotion_id.
        """
        self.daily_promotion_revenue = self.promotions.query("has_promotion == True").groupby(
            ["daily", "promotion_id"]).agg({"payment_amount": 'sum'}).reset_index().rename(
            columns={"payment_amount": "total_revenue"})
        return self.daily_promotion_revenue

    def get_daily_promotion_discount(self):
        """
        Orders which have promotion columns with null values.
        This process indicates total **NOT** promoted discount amount per hour per promotion_id.
        """

        if len(self.promotions.query("discount_amount == discount_amount")) != 0:
            self.daily_promotion_discount = self.promotions.query(
                "has_promotion == True and discount_amount == discount_amount").groupby(["daily", "promotion_id"]).agg(
            {"discount_amount": 'sum'}).reset_index().rename(columns={"discount_amount": "total_discount"})
        return self.daily_promotion_discount

    def get_promotion_average_order_count_per_customer(self):
        """
        average order count per promotion per customer
        """
        average_order_count_per_cust = self.promotions.groupby(["client", "promotion_id"]).agg(
            {"id": lambda x: len(np.unique(x))}).reset_index().rename(columns={"id": "average_order_count"})
        return average_order_count_per_cust.groupby("promotion_id").agg({"average_order_count": "mean"}).reset_index()

    def get_promotion_number_of_customer(self):
        """
        average order count per promotion per customer
        """
        self.promotion_number_of_customer = self.promotions.groupby("promotion_id").agg(
            {"client": lambda x: len(np.unique(x))}).reset_index().rename(columns={"client": "client_count"})
        return self.promotion_number_of_customer

    def get_promotion_kpis(self):
        """
        average daily order count per promotion
        average daily payment amount per promotion
        average daily discount amount per promotion
        number of user per promotion
        """
        kpi_1 = self.inorganic_orders.groupby("promotion_id").agg({"order_count": "mean"}).reset_index()
        kpi_2 = self.daily_promotion_revenue.groupby("promotion_id").agg({"total_revenue": "mean"}).reset_index()
        kpi_3 = self.daily_promotion_discount.groupby("promotion_id").agg({"total_discount": "mean"}).reset_index()
        kpi_4 = self.promotion_number_of_customer

        promotions = pd.DataFrame(list(self.promotions['promotion_id'].unique())).rename(columns={0: "promotion_id"})
        return promotions.merge(kpi_1, on='promotion_id').merge(
            kpi_2, on='promotion_id').merge(kpi_3, on='promotion_id').merge(kpi_4, on='promotion_id').fillna(0)

    def execute_promotion_analysis(self, end_date=None):
        """
            1. Check for promotions data available on elasticsearch orders index.
            2. Execute each analysis separately.
            3. Insert into 'reports' index.
        """
        if self.has_promotion_connection:
            self.get_promotions(end_date=convert_to_iso_format(current_date_to_day() if end_date is None else end_date))
            for ins in list(zip(self.analysis, [self.get_inorganic_orders_per_promotion_per_day,
                                                self.get_organic_orders_per_day,
                                                self.get_inorganic_orders_per_promotion_per_hour,
                                                self.get_daily_inorganic_ratio, self.get_hourly_inorganic_ratio,
                                                self.get_daily_promotion_revenue, self.get_daily_promotion_discount,
                                                self.get_promotion_average_order_count_per_customer,
                                                self.get_promotion_number_of_customer,
                                                self.get_promotion_kpis])):
                _result = ins[1]()
                self.insert_into_reports_index(promotion_analytics=_result,
                                               pa_type=ins[0], start_date=end_date, index=self.order_index)
                del _result

    def insert_into_reports_index(self, promotion_analytics, pa_type, start_date=None, index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": start_date or current date,
         "report_name": "promotion_analytic",
         "index": "main",
         "report_types": {"inorganic_orders_per_promotion_per_day", "daily_organic_orders",
                         "daily_inorganic_ratio", "hourly_inorganic_ratio",
                         "daily_promotion_revenue", "daily_promotion_discount", "avg_order_count_per_promo_per_cust"
                          },
         "data": promotion_analytic.to_dict("results") -  dataframe to list of dictionary
         }
        :param promotion_analytic: data set, data frame
        :param start_date: data start date
        :param pa_type: promotion analytic type
        :param index: dimentionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = [{"id": np.random.randint(200000000),
                        "report_date": current_date_to_day().isoformat() if start_date is None else start_date,
                        "report_name": "promotion_analytic",
                        "index": get_index_group(index),
                        "report_types": {"type": pa_type},
                        "data": promotion_analytics.fillna(0).to_dict("results")}]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def fetch(self, promotion_analytic_name, start_date=None):
        """
        This allows us to query the created Promotion Analytics.
        promotion_analytic_name is crucial for us to collect the correct filters.
        Example of queries;
            -   promotion_analytic_name: inorganic_orders_per_promotion_per_day,
            -   start_date: 2021-01-01T00:00:00

            {'size': 10000000,
            'from': 0,
            '_source': True,
            'query': {'bool': {'must': [
                                        {'term': {'report_name': 'p'}},
                                        {"term": {"index": "orders_location1"}}
                                        {'term': {'report_types.type': 'most_ordered_products'}},
                                        {'range': {'report_date': {'lt': '2021-04-01T00:00:00'}}}]}}}

            - start date will be filtered from data frame. In this example; .query("daily > @start_date")

        :param promotion_analytic_name: most_ordered_products.
        :param start_date: product analytic report created date
        :param index: index_name in order to get dimension_of data. If there is no dimension, no need to be assigned
        :return: data frame
        """
        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_types.type": promotion_analytic_name}},
                           {"term": {"index": get_index_group(self.order_index)}}]

        if start_date is not None:
            date_queries = [{"range": {"report_date": {"lt": convert_to_iso_format(start_date)}}}]

        self.query_es = QueryES(port=self.port,
                                host=self.host)
        self.query_es.query_builder(fields=None, _source=True,
                                    date_queries=date_queries,
                                    boolean_queries=boolean_queries)
        _res = self.query_es.get_data_from_es(index="reports")
        _data = pd.DataFrame()
        if len(_res) != 0:
            _data = pd.DataFrame(_res[0]['_source']['data'])
        return _data
