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

from configs import default_es_port, default_es_host
from utils import current_date_to_day, get_index_group, convert_str_to_hour, convert_to_date, dimension_decision
from data_storage_configurations.query_es import QueryES


class ProductAnalytics:
    """

    """
    def __init__(self,
                 has_product_connection=True,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        !!!!
        ******* ******** *****
        Dimensional Product Analytic:
        Funnels must be created individually for dimensions. For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimension can be assigned in order to create a Product Analyse

        download_index; downloads_location1 this will be the location dimension of
                        parameters in order to query downloads indexes; 'location1'.
        download_index; orders_location1 this will be the location dimension of
                        parameters in order to query orders indexes; 'location1'.
        ******* ******** *****
        !!!

        :param has_product_connection: has_product_connection if there is no products data then, no product analytics
        :param host: elasticsearch host
        :param port: elasticsearch port
        :param download_index: elasticsearch port
        :param order_index: elasticsearch port
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.has_product_connection = has_product_connection
        self.download_index = download_index
        self.order_index = order_index
        self.query_es = QueryES(port=port, host=host)
        self.fields_products = ["id", "client", "basket", "session_start_date", "payment_amount"]
        self.products = pd.DataFrame()
        self.hourly_products = pd.DataFrame()
        self.hourly_product_cat = pd.DataFrame()
        self.product_pairs = pd.DataFrame()
        self.analysis = ["most_ordered_products", "most_ordered_categories", "hourly_products_of_sales",
                         "hourly_categories_of_sales", "hourly_categories_of_sales", "most_combined_products"]

    def dimensional_query(self, boolean_query=None):
        if dimension_decision(self.order_index):
            if boolean_query is None:
                boolean_query = [{"term": {"dimension": self.order_index}}]
            else:
                boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def get_products(self, end_date):
        """
            1.  Fetch data from orders with filter; actions.purchased: True
            2.  Example of basket dictionary on order document;
                'basket': {'p_10': {'price': 6.12, 'category': 'p_c_8', 'rank': 109},
                           'p_145': {'price': 12.0, 'category': 'p_c_9', 'rank': 175},
                           'p_168': {'price': 13.12, 'category': 'p_c_10', 'rank': 82},
                           'p_9': {'price': 0.52, 'category': 'p_c_3', 'rank': 9},
                           'p_4': {'price': 3.72, 'category': 'p_c_8', 'rank': 69},
                           'p_104': {'price': 8.88, 'category': 'p_c_10', 'rank': 97},
                           'p_74': {'price': 8.395, 'category': 'p_c_10', 'rank': 35}
                           }
                a. Keys of dictionary are products.
        :param end_date: last date of data set
        """
        self.query_es = QueryES(port=self.port, host=self.host)
        self.query_es.date_queries_builder({"session_start_date": {"lt": end_date}})
        self.query_es.query_builder(fields=None, _source=True,
                                    boolean_queries=self.dimensional_query([{"term": {"actions.has_basket": True}}]))
        self.products = self.query_es.get_data_from_es()
        self.products = pd.DataFrame([{col: r['_source'][col] for col in self.fields_products} for r in self.products])
        self.products = self.products.query('basket == basket')
        self.products['products'] = self.products['basket'].apply(
            lambda x: list(x.keys()) if x == x else None)  # get product_ids
        self.products = self.products.query('products == products')
        # get prices
        self.products['price'] = self.products.apply(
            lambda row: [row['basket'][i]['price'] for i in row['products']], axis=1)

    def get_most_ordered_products(self):
        """
        the products of orders are counted and sorted according to the order counts.
        So, the most ordered products are detected.
        """
        return self.products.groupby("product").agg(
            {"id": lambda x: len(np.unique(x))}).reset_index().rename(
            columns={"id": "order_count"}).sort_values(by='order_count', ascending=False)

    def get_most_ordered_categories(self):
        """
        the product categories of orders are counted and sorted according to the order counts.
        So, the most ordered products are detected.
        """
        return self.products.groupby("category").agg({"id": lambda x: len(np.unique(x))}).reset_index().rename(
            columns={"id": "order_count"}).sort_values(by='order_count', ascending=False)

    def get_hourly_products_of_sales(self):
        """
        Which product is preferred to purchase via purchased hour?
        """
        self.products['hours'] = self.products['session_start_date'].apply(lambda x: convert_str_to_hour(x))
        self.hourly_products = self.products.groupby(["hours", "product"]).agg(
            {"id": lambda x: len(np.unique(x))}).reset_index().rename(columns={"id": "order_count"})
        self.hourly_products['product_order_count_rank'] = self.hourly_products.sort_values(by=["hours", "order_count"],
                                                                                            ascending=False).groupby(
            "hours").cumcount() + 1
        self.hourly_products = pd.merge(self.hourly_products,
                                        self.products.groupby("hours").agg({"id": lambda x: len(np.unique(x))}
                                                                           ).reset_index().rename(
                                            columns={"id": "hourly_total_orders"}), on='hours', how='left')
        self.hourly_products['hourly_order_ratio'] = self.hourly_products['order_count'] / self.hourly_products[
            'hourly_total_orders']
        return self.hourly_products

    def get_hourly_categories_of_sales(self):

        self.hourly_product_cat = self.products.groupby(["hours", "category"]).agg(
            {"id": lambda x: len(np.unique(x))}).reset_index().rename(columns={"id": "order_count"})
        self.hourly_product_cat['product_cat_order_count_rank'] = self.hourly_product_cat.sort_values(by=["hours", "order_count"],
                                                                                            ascending=False).groupby(
            "hours").cumcount() + 1
        self.hourly_product_cat = pd.merge(self.hourly_product_cat,
                                      self.products.groupby("hours").agg({"id": lambda x: len(np.unique(x))}
                                                                    ).reset_index().rename(
                                          columns={"id": "hourly_total_orders"}),
                                      on='hours', how='left')
        self.hourly_product_cat['hourly_order_ratio'] = self.hourly_product_cat['order_count'] / self.hourly_product_cat[
            'hourly_total_orders']
        return self.hourly_product_cat

    def get_most_combined_products(self):
        self.product_pairs = self.products.groupby("id").agg({"product": lambda x: list(x)}).reset_index()
        self.product_pairs['product_pairs'] = self.product_pairs['product'].apply(
            lambda x: np.array(list(product(x, x))).tolist())
        self.product_pairs['total_pairs'] = 1
        self.product_pairs = self.product_pairs[self.product_pairs[0] != self.product_pairs[1]]
        self.product_pairs = self.product_pairs.groupby([0, 1]).agg({"total_pairs": "sum"}).reset_index()
        self.product_pairs = self.product_pairs.sort_values('total_pairs', ascending=False)
        return self.product_pairs

    def execute_product_analysis(self, start_date=None):
        if self.has_product_connection:
            self.get_products()
            for ins in list(zip(self.analysis, [self.get_most_ordered_products, self.get_most_ordered_categories,
                                                self.get_hourly_products_of_sales, self.get_hourly_categories_of_sales,
                                                self.get_most_combined_products])):
                _result = ins[1]()
                self.insert_into_reports_index(product_analytic=_result,
                                               pa_type=ins[0], start_date=start_date, index=self.order_index)

    def insert_into_reports_index(self, product_analytic, pa_type, start_date=None, index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": start_date or current date,
         "report_name": "product_analytic",
         "index": "main",
         "report_types": {"type": "most_ordered_products", "most_ordered_categories", "hourly_products_of_sales",
                            "hourly_categories_of_sales", "hourly_categories_of_sales", "most_combined_products"
                          },
         "data": product_analytic.to_dict("results") -  dataframe to list of dictionary
         }
        :param product_analytic: data set, data frame
        :param start_date: data start date
        :param pa_type: product analytic type
        :param index: dimentionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = [{"id": np.random.randint(200000000),
                        "report_date": current_date_to_day().isoformat() if start_date is None else start_date,
                        "report_name": "product_analytic",
                        "index": get_index_group(index),
                        "report_types": {"type": pa_type},
                        "data": product_analytic.fillna(0).to_dict("results")}]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def fetch(self, product_analytic_name, start_date=None):
        """
        This allows us to query the created Product Analytics.
        product_analytic_name is crucial for us to collect the correct filters.
        Example of queries;
            -   product_analytic_name: most_ordered_products,
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

        :param funnel_name: funnel the whole name, includes funnel type and time period.
        :param start_date: funnel first date
        :param end_date: funnel last date
        :param index: index_name in order to get dimension_of data. If there is no dimension, no need to be assigned
        :return: data frame
        """



        report_name, funnel_type, time_period = funnel_name.split("_")
        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_name": report_name}},
                           {"term": {"index": get_index_group(self.order_index)}},
                           {"term": {"report_types.time_period": time_period}},
                           {"term": {"report_types.type": funnel_type}}]

        if end_date is not None:
            date_queries = [{"range": {"report_date": {"lt": convert_to_iso_format(end_date)}}}]

        self.query_es = QueryES(port=self.port,
                                host=self.host)
        self.query_es.query_builder(fields=None, _source=True,
                                    date_queries=date_queries,
                                    boolean_queries=boolean_queries)
        _res = self.query_es.get_data_from_es(index="reports")
        _data = pd.DataFrame()
        if len(_res) != 0:
            _data = pd.DataFrame(_res[0]['_source']['data'])
            if start_date is not None:
                if time_period not in ['yearly', 'hourly']:
                    _data[time_period] = _data[time_period].apply(lambda x: convert_to_date(x))
                    start_date = convert_to_date(start_date)
                    _data = _data[_data[time_period] >= start_date]
        return _data
