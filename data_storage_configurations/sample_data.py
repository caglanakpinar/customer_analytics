import numpy as np
import pandas as pd
import datetime
import random
from time import gmtime, strftime
import pytz
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import argparse

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from configs import elasticsearch_settings, elasticsearch_settings_downloads, default_sample_data_previous_day
from utils import current_date_to_day


class CreateSampleIndex:
    def __init__(self, host, port, start_date, end_date, prev_day_count):
        """
        Creating Sample orders and downloads ElasticSearch Index:
            - Sample indexes must be predictable.
            - It must allow us to interpret for Exploratory Analysis.
            - It can be useful for generating dashboards and charts, in order to impress users.
            - Sample Indexes must be realistic.
            - This aims to create an ElasticSearch index with data
              which will be similar to any e-commerce business of data.
            - Data sets are created with actions of e-commerce of the purchase process and users acquisition processes;
                - Download
                - Signup
                - Login
                - product add to basket
                - Order screen
                - Purchase

            Example of Order (key, value dictionary);
             {'id': 66597918,
               'date': '2020-12-13T09:20:00',
               'actions': {'has_sessions': True,
                'has_basket': True,
                'order_screen': False,
                'purchased': False,
                'add_to_basket': {'p_28': {'duration': 42, '_date': '2020-12-13T09:16:48'},
                 'p_45': {'duration': 23, '_date': '2020-12-13T09:17:11'},
                 'p_135': {'duration': 26, '_date': '2020-12-13T09:17:37'},
                 'p_144': {'duration': 95, '_date': '2020-12-13T09:19:12'},
                 'p_59': {'duration': 26, '_date': '2020-12-13T09:19:38'},
                 'p_71': {'duration': 22, '_date': '2020-12-13T09:20:00'}}},
               'client': 'u_552873',
               'promotion_id': None,
               'payment_amount': 20.725,
               'discount_amount': 0,
               'basket': {'p_28': {'price': 3.0, 'category': 'p_c_7', 'rank': 1},
                'p_45': {'price': 9.375, 'category': 'p_c_9', 'rank': 92},
                'p_135': {'price': 0.255, 'category': 'p_c_1', 'rank': 188},
                'p_144': {'price': 4.095, 'category': 'p_c_6', 'rank': 70},
                'p_59': {'price': 2.32, 'category': 'p_c_1', 'rank': 15},
                'p_71': {'price': 1.68, 'category': 'p_c_3', 'rank': 155}},
               'total_products': 6,
               'session_start_date': '2020-12-13T09:16:06'}

        There are some default expected values. Some are stored at configs.py,
        some are stored as constant directly into the code in order to see the impact of creating the data.
        :param host: default 'localhost'
        :param port: default ElasticSearch default port 9200
        :param start_date: date aims to start to create sample data default start date is at configs.py
        :param end_date: date aim to end to create sample data default end date is the recent date.
        :param prev_day_count: number of days back for data creation
        """
        self.host = host if host is not None else '127.0.0.1'
        self.port = int(port) if host is not None else 9200
        self.index = 'orders'
        self.start_date = start_date
        self.end_date = end_date
        self.es_settings = elasticsearch_settings
        self.es_settings_downloads = elasticsearch_settings_downloads
        self.es = None
        self.prev_day_count = prev_day_count
        self.current_date = current_date_to_day()
        self.days = []
        self.weeks = []
        self.weekly_customers = []
        self.orders = []
        self.client_active = None
        self.customer_newcomer_ratio = None
        self.customer_newcomer_list = None
        self.customer_churn_ratio = None
        self.customer_churn_list = None
        self._newcomer_list = None
        self.products_list = None
        self.products_lis = None
        self.products_category_list = None
        self.products = None
        self.products_list = None
        self.order_of_num_products = None
        self.order_of_num_products_for_no_orders = None
        self.product_similarity_ratio_when_has_order_but_has_uncomplete_order = None
        self.days_part_of_total_orders = None
        self.promotion_list = None
        self.promo_levels = None
        self.discounts = None
        self.promo_categories = None
        self.promotions = None
        self.has_promo = None
        self.has_newcomer_promo = None
        self.action_funnel_related_to_order = None
        self.client_has_session_before_order_min = None
        self.has_seen_basket_after_session = None
        self.order_duration_from_session_to_purchase = None
        self.orders_df = pd.DataFrame()
        self.orders_df_pv = pd.DataFrame()
        self.downloads_df = pd.DataFrame()
        self.days_part_of_total_downloads = None
        self.rest_of_clients = None
        self.customers_orders_download_diff = None

    def connect_elastic_search(self):
        """
        connect to ElasticSearch

        """
        self.es = Elasticsearch([{'host': self.host, 'port': self.port}], timeout=3000, max_retries=100)

    def create_index(self):
        """
        creating ElasticSearch Indexes. These are 'orders' and 'downloads'
        If these have already been created, skip processes.
        """
        try:
            self.es.indices.create('orders', body=self.es_settings)
        except Exception as e:
            print("index already exists !!!")

        try:
            self.es.indices.create('downloads', body=self.es_settings_downloads)
        except Exception as e:
            print("index already exists !!!")

    def decide_date(self, date, start=True):
        """
        Decides the date start or end date of the randomly sampling data.
        If one of each (start, or end) has not been assigned default_sample_data_previous_day form configs.py works.
        :param date: date format for assigning start or end date date decision
        :param start: which date we are deciding on? start or end date?
        :return: date format: datetime
        """
        if date is None:
            if start:
                if self.prev_day_count is not None:
                    date = datetime.datetime.now() - datetime.timedelta(days=self.prev_day_count)
                else:
                    date = datetime.datetime.now() - datetime.timedelta(days=default_sample_data_previous_day)
            else:
                date = datetime.datetime.now()
        else:
            date = datetime.datetime.strptime(str(date)[0:10] + ' 00:00:00', '%Y-%m-%d %H:%M:%S')

        return date

    def create_dates(self):
        """
        creates dates iterating from the start date to the end date.
        Randomly Sampling is working day by day.
        We also need weeks. At here we are assigning weeks as Mondays of each week.
        """
        self.start_date = self.decide_date(self.start_date)
        self.end_date = self.decide_date(self.end_date, start=False)

        while self.start_date < self.end_date:
            next_date = self.start_date + datetime.timedelta(days=1)
            self.days += [(self.start_date, next_date)]
            if self.start_date.isoweekday() == 1:
                self.weeks += [self.start_date]
            self.start_date += datetime.timedelta(days=1)

    def create_customers(self):
        """
        -   Customers are randomly created with the format 'u_2000'. Each iteration randomly users are selected.
        -   There is a static newcomer ratio here. Each day plus the order count, there must be the ratio of newcomers.
        -   There is also the 'customer_churn_list' function which
        allows us to calculate the randomly selected churn rate for each day and day part.
        """
        self.client_active = ['u_' + str(i) for i in range(1000000)]
        self.customer_newcomer_ratio = np.arange(0, 0.3, 0.025).tolist()
        self.customer_newcomer_list = lambda ratio, active_count: np.random.randint(1000000, 2000000, size=int(
            active_count * ratio)).tolist()
        self.customer_churn_ratio = np.arange(0.2, 0.6, 0.025).tolist()
        self.customer_churn_list = lambda ratio, active_count, active_list: random.sample(active_list, size=int(
            active_count * ratio))
        self._newcomer_list = ['u_' + str(i) for i in
                          self.customer_newcomer_list(random.sample(self.customer_newcomer_ratio, 1)[0], 13000)]

    def create_product_list(self):
        """
            -   Products are optional when it is created the original index.
                However, here just for an example, it creates the products.
            -   Products are assigned as 'p_100'
            -   Product Categories are assigned as 'p_c_100'.
            -   Each product have their own prices related to their categories.
                'price_multiplier' are also decided related to categories of purchasability.
            -   'ratio' is helping us to a ratio of the product of choosability of the customers.

        """
        products_list = ['p_' + str(i) for i in range(200)]
        self.products_category_list = {}
        for i in range(1, 11):
            _cat = 'p_c_' + str(i)
            self.products_category_list[_cat] = {}
            if i in [9, 10]:
                self.products_category_list[_cat]['price_multiplier'] = list(range(20, 35))
                self.products_category_list[_cat]['ratio'] = 0.2
            if i in range(5, 9):
                self.products_category_list[_cat]['price_multiplier'] = list(range(10, 21))
                self.products_category_list[_cat]['ratio'] = 0.5
            if i in range(1, 6):
                self.products_category_list[_cat]['price_multiplier'] = list(range(1, 11))
                self.products_category_list[_cat]['ratio'] = 0.3

        a = list(range(len(products_list)))
        random.shuffle(a)

        self.products = {}
        for ratio in [0.2, 0.5, 0.3]:
            _p_cats = [i for i in self.products_category_list if self.products_category_list[i]['ratio'] == ratio]
            _products = random.sample(products_list, int(len(products_list) * ratio)) if ratio != 0.3 else products_list
            products_list = list(set(products_list) - set(_products))
            for p in _products:
                _p_category = random.sample(_p_cats, 1)[0]
                _multiplier = random.sample(self.products_category_list[_p_category]['price_multiplier'], 1)[0]
                self.products[p] = {"category": _p_category,
                                    "price": 0.5 * round((0.5 * _multiplier) + (pow(_multiplier, 2) / 100), 2)
                               }
        counter = 0
        for p in self.products:
            self.products[p]['rank'] = a[counter]
            counter += 1
        self.products_list = ['p_' + str(i) for i in range(200)]

    def create_order_product_count(self):
        """
        The number of product count in order. These number (integer) is randomly selected for each order or basket.
        """
        self.order_of_num_products = list(range(3, 5)) * 30 + list(range(5, 9)) * 50 + list(range(9, 11)) * 20
        self.order_of_num_products_for_no_orders = [0] * 50 + list(range(1, 3)) * 50 + list(range(3, 5)) * 30 + list(
            range(5, 9)) * 50 + list(range(9, 11)) * 20
        self.product_similarity_ratio_when_has_order_but_has_uncomplete_order = [0] * 20 + [0.5] * 40 + [0.75] * 30 + [
            1] * 10

    def create_day_part_order_counts(self):
        """
        The number of Purchased Order Count per day part per week/weekend:
        Each day or daypart is sampling individually. However, these dayparts differ related to weekend or weekday.
        Each day part can have a range of numbers which indicates
        the total number of orders for queried week part.
        """
        self.days_part_of_total_orders = {'week':
                                               {"nights":
                                                    {"range": np.arange(100, 200, 1).tolist(),
                                                        "hours": [0, 8]},
                                                "mornings":
                                                  {"range": np.arange(1000, 1500, 1).tolist(),
                                                   "hours": [8, 18]},
                                                "evenings":
                                                   {"range": np.arange(2200, 4200, 1).tolist(),
                                                    "hours": [18, 24]}},
                                          'weekend':
                                                  {"nights":
                                                    {"range": np.arange(100, 200, 1).tolist(),
                                                     "hours": [1, 9]},
                                                     "mornings":
                                                     {"range": np.arange(3000, 5000, 1).tolist(),
                                                      "hours": [9, 16]},
                                                     "evenings":
                                                     {"range": np.arange(2800, 4500, 1).tolist(),
                                                    "hours": [16, 24]}
                                                            }
                                          }

    def create_promotions(self):
        """
        Promotions are also optional for the original 'orders' index.
        Promotions directly affect the discount rate in the sampling methodology.
        There are 4 product categories;
            -   'acquisition',
            -   'engaged',
            -   'upsell_basket',
            -   'active'
        Each category has a ratio that refers to the discount ratio of the order.
        If the discount order is higher than the other promotion categories,
         the number of orders related to that promotions must be lower than that promotion.
        """
        self.promotion_list = ['promo_' + str(i) for i in range(100)]
        self.promo_levels = {}

        self.discounts = {r: np.arange(max(0.01, 1 - r - 0.8), 1 - r - 0.5, 0.05).tolist() for r in [0.05, 0.2, 0.3, 0.45]}
        self.promo_categories = {_p[0]: _p[1] for _p in
                            zip([0.05, 0.2, 0.2, 0.3, 0.3], ['acquisition', 'engaged', 'upsell_basket', 'active'])}

        self.promotions = {}
        for ratio in [0.05, 0.2, 0.2, 0.3, 0.3]:
            _promos = random.sample(self.promotion_list,
                                    int(len(self.promotion_list) * ratio)) if ratio != 0.45 else self.promotion_list
            self.promotion_list = list(set(self.promotion_list) - set(_promos))
            for p in _promos:
                self.promotions[p] = {"category": self.promo_categories[ratio],
                                      "discount_rate": random.sample(self.discounts[ratio], 1)[0]}

        self.has_promo = [True] * 40 + [False] * 60
        self.has_newcomer_promo = [True] * 80 + [False] * 20

    def order_basket_duration(self, product):
        """
        Action which is 'add to basket' is also optional.
        however, in the sampling, 'each adds to basket' actions are calculated.
        :param product: each product of purchasability ratio will differ
                        the duration of customer decision of adding for purchase.
        :return: duration (sec)
        """
        if self.products_category_list[self.products[product]['category']]['ratio'] == 0.2:
            duration = random.sample(list(range(10, 200)), 1)[0]
        if self.products_category_list[self.products[product]['category']]['ratio'] == 0.5:
            duration = random.sample(list(range(10, 100)), 1)[0]
        if self.products_category_list[self.products[product]['category']]['ratio'] == 0.3:
            duration = random.sample(list(range(10, 30)), 1)[0]
        return duration

    def purchase_to_basket_duration(self, average_amount, total_amount):
        """
        Calculation of total duration per product related to add to basket action.
        According to the total amount of the basket increases related to products of price, the duration will change.
        If there is a higher price, the duration will be higher too.
        :param average_amount: Average basket value when a product is added to the basket.
        :param total_amount: Total basket value when a product is added to the basket.
        :return: Duration (sec)
        """
        random_duration = random.sample(list(range(20, 120)), 1)[0]
        multiplier = (total_amount - average_amount) / max(total_amount, average_amount)
        additional_duration = random_duration * multiplier * 0.05
        return random_duration + additional_duration

    def get_actions(self):
        """
        Actions are optional. But, 'has_sessions' and 'purchased' actions are required.
        Ratio ranges of an order of steps from session to purchase.
        Actions;
            - has sessions     : Boolean; indicates a login process of a client
            - has has_basket   : Boolean; indicates a 'add to basket' process of a client
            - has order_screen : Boolean; indicates a 'payment screen' process of a client
            - has purchased    : Boolean; indicates an order process of a client
        """
        actions = ['has_sessions', 'has_basket', 'order_screen', 'purchased']
        self.action_funnel_related_to_order = {'has_sessions': np.arange(1.5, 2, 0.01).tolist(),
                                               'has_basket': np.arange(1.3, 1.5, 0.01).tolist(),
                                               'order_screen': np.arange(1.3, 1.5, 0.01).tolist(),
                                               'purchased': 1
                                              }
        for a in actions:
            self.action_funnel_related_to_order[a] = np.arange(1.5, 2, 0.01).tolist()
        self.client_has_session_before_order_min = [0] * 1200 + np.arange(1, 60, 1).tolist() * 3 + np.arange(60, 120,
                                                                                                            1).tolist() * 2 + np.arange(
            60, 120, 1).tolist() * 2
        self.has_seen_basket_after_session = [False] * 75 + [True] * 25
        self.order_duration_from_session_to_purchase = lambda product_count: random.sample(list(range(10, 200), 1)[0])

    def get_download_hour_diff(self):
        """
        Calculating customer's first order or session to download the hour difference.

        """
        self.customers_orders_download_diff = {}
        for i in range(0, 1000):
            if i in [1, 2, 3]:
                self.customers_orders_download_diff[i] = list(range(5, 200)) * 50 + \
                                                         list(range(200, 400)) * 30 + list(range(400, 500)) * 20
            if i in list(range(4, 10)):
                self.customers_orders_download_diff[i] = list(range(5, 120)) * 50 + \
                                                         list(range(120, 240)) * 30 + list(range(240, 360)) * 20
            if i in list(range(10, 2000)):
                self.customers_orders_download_diff[i] = list(range(5, 72)) * 50 + \
                                                         list(range(72, 144)) * 30 + list(range(144, 2216)) * 20

    def get_download_order_ratios(self):
        """
        The number of Download Count related to Order Count per day part per week/weekend:
        Each day or dayparts are sampling individually. However, these dayparts differ related to weekend or weekday.
        Each day part and order can have a range of numbers which indicates
        the total number of downloads for the queried week part.

        """
        self.days_part_of_total_downloads = {'week':
                                              {"nights":
                                                   {"range": np.arange(1.1, 1.3, 0.1).tolist(),
                                                    "hours": [0, 8]},
                                               "mornings":
                                                   {"range": np.arange(1.1, 1.8, 0.1).tolist(),
                                                    "hours": [8, 18]},
                                               "evenings":
                                                   {"range": np.arange(1.1, 2.8, 0.1).tolist(),
                                                    "hours": [18, 24]}},
                                          'weekend':
                                              {"nights":
                                                   {"range": np.arange(1.1, 1.3, 0.1).tolist(),
                                                    "hours": [1, 9]},
                                               "mornings":
                                                   {"range": np.arange(1.1, 2.2, 0.1).tolist(),
                                                    "hours": [9, 16]},
                                               "evenings":
                                                   {"range": np.arange(1.1, 3.0, 0.1).tolist(),
                                                    "hours": [16, 24]}
                                               }
                                          }

    def get_day_part(self, hour, is_weekend):
        """
        day part decision at download are sampled.
        :param hour: realted hour
        :param is_weekend: week (isoweekday; 1, 2, 3, 4, 5) or weekend (iso-weekday; 6, 7)
        :return: nights, mornings, evenings
        """
        if is_weekend == 'week':
            if 0 <= hour < 9:
                return 'nights'
            if 9 <= hour < 18:
                return 'mornings'
            if 18 <= hour < 24:
                return 'evenings'
        if is_weekend == 'weekend':
            if 0 <= hour < 9:
                return 'nights'
            if 9 <= hour < 16:
                return 'mornings'
            if 16 <= hour < 24:
                return 'evenings'

    def get_customer_order_to_detected_downloads(self):
        """
        Downloads Index Sampling:
            - Sampling starts after the order index sampling process is done.
            - Each download transaction is processed by each client. So, client and download numbers must be the same.
            - First, 1. Clients who have at least one purchased transaction, are collected.
                    2.  Clients who have no order but has session transaction, are collected.
            - Both client list of download dates are sampled individually.
            - The download date is formed per day per daypart for each client.
            - When the 'downloads' index is created with the download date,
              signup date is also creating a date between the download date and first session date.
        """
        self.connect_elastic_search()
        self.create_index()
        self.create_dates()
        self.create_customers()
        self.create_product_list()
        self.create_order_product_count()
        self.create_day_part_order_counts()
        self.create_promotions()
        self.get_actions()
        self.get_download_hour_diff()
        self.get_download_order_ratios()

        print("query date :", self.days[0][0].isoformat())
        match = {"size": 1000000, "from": 0,
                 "query": {
                      "bool": {
                          "filter": {"range": {"session_start_date": {"gte": self.days[0][0].isoformat()}}}
                      }
                  },
                  "fields": ["id", "session_start_date", "client", "actions.purchased"],
                  "_source": True
                  }
        res = []
        for r in self.es.search(index='orders', body=match)['hits']['hits']:
            res.append({
                        'id': r['fields']['id'][0],
                        'date': r['fields']['session_start_date'][0],
                        'client': r['fields']['client'][0],
                        'purchased': 1 if r['fields']['actions.purchased'][0] else 0
                        })
        print("orders are collected !!!!")
        self.orders_df = pd.DataFrame(res)
        print(self.orders_df.head())
        self.orders_df['date'] = self.orders_df['date'].apply(
            lambda x: datetime.datetime.strptime(str(x)[0:10] + ' ' + str(x)[11:19], '%Y-%m-%d %H:%M:%S'))
        self.orders_df['hours'] = self.orders_df['date'].apply(lambda x: x.hour)
        self.orders_df['is_weekend'] = self.orders_df['date'].apply(
            lambda x: 'week' if x.isoweekday() in list(range(1, 6)) else 'weekend')
        self.orders_df['day_parts'] = self.orders_df.apply(
            lambda row: self.get_day_part(row['hours'], row['is_weekend']), axis=1)
        self.orders_df['day_str'] = self.orders_df['date'].apply(lambda x: str(x)[0:10])
        self.orders_df_pv = self.orders_df.groupby("client").agg({"id": "count", "date": "min"}
                                                              ).reset_index().rename(columns={"id": "session_count"})
        self.orders_df_pv = pd.merge(self.orders_df_pv,
                                     self.orders_df.query("purchased == 1").groupby("client").agg(
                                         {"id": "count"}).reset_index().rename(columns={"id": "order_count"}),
                                     on='client', how='left')
        self.orders_df_pv['day_str'] = self.orders_df['date'].apply(lambda x: str(x)[0:10])

        self.orders_df = self.orders_df.groupby(["day_parts",
                                                 "is_weekend",
                                                 "day_str"]).agg({"id": "count"}).reset_index().rename(
            columns={"id": "session_count"})

    def get_insert_obj(self, list_of_obj, index):
        """
        bulk insert list is creating.
        :param list_of_obj: list of objects (orders)
        :param index: downloads or orders
        """
        for i in list_of_obj:
            add_cmd = {"_index": index,
                       "_source": i}
            yield add_cmd

    def get_ordered_users_of_downloads(self):
        """
        Downloads Index Sampling:
            - Sampling starts after the order index sampling process is done.
            - Each download transaction is processed by each client. So, client and download numbers must be the same.
            - First, 1. Clients who have at least one purchased transaction, are collected.
                    2.  Clients who have no order but has session transaction, are collected.
            - Both client list of download dates are sampled individually.
            - The download date is formed per day per daypart for each client.
            - When the 'downloads' index is created with the download date,
              signup date is also creating a date between the download date and first session date.
        """
        downloads = []
        for c in self.orders_df_pv.to_dict("results"):
            _download_obj = {"id": None, "download_date": None, "signup_date": None, 'client': None}
            _date_diff = random.sample(self.customers_orders_download_diff[c['session_count']], 1)[0]
            _date = c['date'] - datetime.timedelta(hours=_date_diff)
            _date_signup = c['date'] - datetime.timedelta(minutes=1)
            _download_obj["id"] = np.random.randint(200000000)
            _download_obj["download_date"] = _date
            _download_obj["signup_date"] = _date_signup
            _download_obj["client"] = c['client']
            downloads.append(_download_obj)
        print("number of download insert :",  len(downloads))
        helpers.bulk(self.es, self.get_insert_obj(downloads, 'downloads'))
        self.downloads_df = pd.DataFrame(downloads)
        self.rest_of_clients = list(set(self.client_active) - set(list(self.orders_df_pv['client'])))
        print("rest of clients :", len(self.rest_of_clients))
        del _download_obj

    def get_downloads(self):
        """
        Downloads Index Sampling:
            - Sampling starts after the order index sampling process is done.
            - Each download transaction is processed by each client. So, client and download numbers must be the same.
            - First, 1. Clients who have at least one purchased transaction, are collected.
                    2.  Clients who have no order but has session transaction, are collected.
            - Both client list of download dates are sampled individually.
            - The download date is formed per day per daypart for each client.
            - When the 'downloads' index is created with the download date,
              signup date is also creating a date between the download date and first session date.
        """
        self.get_customer_order_to_detected_downloads()
        self.get_ordered_users_of_downloads()
        print(self.es.cat.count('orders', params={"format": "json"}))
        query_str = "day_parts == @day_part and day_str == @_day_str"
        for d in self.days:
            downloads = []
            sessions = []
            _start_date, _day_str = d[0], str(d[0])[0:10]
            week_part = 'week' if d[0].isoweekday() in range(6) else 'weekend'
            import numpy as np
            for day_part in ["nights", "mornings", "evenings"]:
                _orders_df = self.orders_df.query(query_str)
                if len(_orders_df) != 0:
                    _orders = list(_orders_df['session_count'])[0]
                    total_downloads = int((_orders * random.sample(
                        self.days_part_of_total_downloads[week_part][day_part]['range'], 1)[0]) - _orders)
                    print(total_downloads, _orders, len(self.rest_of_clients))
                    if total_downloads > 0 and len(self.rest_of_clients) > int(total_downloads):
                        _rest_of_clients = random.sample(self.rest_of_clients, int(total_downloads))
                        print("rest of clients :", len(_rest_of_clients))
                        hour_range = self.days_part_of_total_orders[week_part][day_part]["hours"]
                        mins = list(range((hour_range[1] - hour_range[0]) * 60))
                        _date = _start_date - datetime.timedelta(hours=hour_range[0])
                        print("date :", _start_date, " ||day part :", hour_range[0])
                        for c in _rest_of_clients:
                            _date_2 = _date + datetime.timedelta(minutes=random.sample(mins, 1)[0])
                            download_date = _date_2.isoformat()
                            _signup_min = random.sample([0] * 700 + list(range(1, 15)) * 5, 1)[0]
                            signup_date = None
                            if _signup_min != 0:
                                signup_date = _date_2 + datetime.timedelta(minutes=_signup_min)
                                _ses_date = signup_date + datetime.timedelta(minutes=1)
                                signup_date = signup_date.isoformat()
                                _ses_obj = {"id": np.random.randint(200000000),
                                            "date": _ses_date.isoformat(),
                                            "actions": {'has_sessions': True,
                                                        'has_basket': False,
                                                        'order_screen': False,
                                                        'purchased': False},
                                            'client': c,
                                            "promotion_id": None,
                                            "payment_amount": None,
                                            "discount_amount": 0,
                                            "basket": {},
                                            "total_products": None,
                                            'session_start_date': _ses_date.isoformat()
                                            }
                                total_products = random.sample(self.order_of_num_products_for_no_orders, 1)[0]
                                if total_products != 0:
                                    _ses_obj["actions"]["has_basket"] = True
                                    _ses_obj['total_products'] = total_products
                                    _products = random.sample(self.products_list, _ses_obj['total_products'])
                                    _ranks = [self.products[p]['rank'] for p in _products]
                                    _products = [p[1] for p in sorted(zip(_ranks, _products))]
                                    _payment_amount = 0
                                    similar_products = []
                                    similar_products_count = min(int(
                                        random.sample(
                                            self.product_similarity_ratio_when_has_order_but_has_uncomplete_order, 1)[
                                            0] * len(_products)), len(_products))
                                    if similar_products_count != 0:
                                        similar_products = random.sample(_products, similar_products_count)
                                    _products = random.sample(self.products_list,
                                                              len(_products) - len(similar_products)) + similar_products
                                    # basket
                                    _payment_amount = 0
                                    _baskets = {}
                                    _duration = 0
                                    _actions = []
                                    for p in _products:
                                        _baskets[p] = {"price": self.products[p]['price'],
                                                       "category": self.products[p]['category'],
                                                       "rank": self.products[p]['rank']}
                                        _payment_amount += self.products[p]['price']
                                        _dur = self.order_basket_duration(p)
                                        _actions.append({'product': p, 'duration': _dur})
                                        _duration += _dur
                                    _start_date = _ses_date - datetime.timedelta(seconds=_duration)
                                    _ses_obj['basket'] = _baskets
                                    _ses_obj['payment_amount'] = _payment_amount
                                    _ses_obj['session_start_date'] = _start_date.isoformat()
                                    # order actions
                                    _ses_obj["actions"]['add_to_basket'] = {}
                                    for _d in _actions:
                                        _start_date += datetime.timedelta(seconds=_d['duration'])
                                        _ses_obj["actions"]['add_to_basket'][_d['product']] = {
                                            'duration': _d['duration'],
                                            '_date': datetime.datetime.strptime(
                                                str(_start_date)[0:19],
                                                '%Y-%m-%d %H:%M:%S').isoformat()
                                            }
                                    if random.sample(self.has_seen_basket_after_session, 1)[0]:
                                        _ses_obj["actions"]["order_screen"] = True
                                sessions.append(_ses_obj)
                                helpers.bulk(self.es, self.get_insert_obj([_ses_obj], 'orders'))
                            downloads.append({"id": np.random.randint(200000000),
                                              "download_date": download_date,
                                              "signup_date": signup_date,
                                              'client': c})
                        self.rest_of_clients = list(set(self.rest_of_clients) - set(_rest_of_clients))

            print(self.es.cat.count('downloads', params={"format": "json"}))
            if len(downloads) != 0:
                print("number of download insert :", len(downloads))
                print(downloads[0:10])
                helpers.bulk(self.es, self.get_insert_obj(downloads, 'downloads'))
            if len(sessions) != 0:
                print("number of sessions insert :", len(sessions))
                print([obj['id'] for obj in sessions[0:10]])
                helpers.bulk(self.es, self.get_insert_obj(sessions, 'orders'))
            print(self.es.cat.count('orders', params={"format": "json"}))
            print(self.es.cat.count('downloads', params={"format": "json"}))

    def execute_sampling(self):
        """
        Orders Index Sampling:
            - Iteratively, from the active client list there randomly the number of clients are sampled
              related to daypart and week/weekend per day.
            - Number of sampled clients must be the number of orders
              which is also randomly selected from range order count list related to
              'days_part_of_total_orders[week_part][day_part]["range"]'.

            - randomly selected client list iteratively converting to a purchased order.
              Each randomly selected active client has an order with actions;
                'has_Sessions', 'has_basket', 'has_order_screeen'.
              When a randomly selected active client of order is created,
                - basket obj is creating; (list of obj);
                     * Number or item in the basket is randomly selected.
                     * According to the number of Number or item in the basket,
                       each obj represents the products that have been purchased from the related client.
                     * When the basket is created, the duration between each product of selection from the client also randomly chosen.
                     * Actions are also created when products are created with the date according to chosen durations.
                - Actions obj is created (list of obj);
                     * Each action must be in the boolean format (True/False).
                       In addition to that add_to_basket_Action is created as a list of objects.

            - Start date and date of each order also calculated in the range of hours of daypart.
            - Purchase Amount is calculated total price of products.
              Each product is also chosen randomly according to the ratio of them.
            - Promotion Orders are also selected randomly with 'has_promo'.
            - Related to the selected promotion, the discount amount is calculated.
              If promotion is None, discount amount = 0.0.

            - Previous Sessions of Each randomly selected active list client;
                * It is possible that users can have multiple times sessions before they have ordered.
                  It is also randomly decided whether or not they have a session before order.
                  This session will be created during the day in the range of daypart.

            - Session with no Orders;
                * It is possible that there might be a number of sessions with no purchase are accomplished.
                * The number of the session with no purchase is also selected randomly.
                * Other actions 'has_basket', 'has _order_screen' are also decided randomly.

            - In the end, we are aiming to create an order index with the session, has_basket, has_order_screen, purchased.

                # of has_Session >= # of has_basket >= # of has_order_screen >= # of purchased

            Example;
            
              {'id': 66597918,
               'date': '2020-12-13T09:20:00',
               'actions': {'has_sessions': True,
                'has_basket': True,
                'order_screen': False,
                'purchased': False,
                'add_to_basket': {'p_28': {'duration': 42, '_date': '2020-12-13T09:16:48'},
                 'p_45': {'duration': 23, '_date': '2020-12-13T09:17:11'},
                 'p_135': {'duration': 26, '_date': '2020-12-13T09:17:37'},
                 'p_144': {'duration': 95, '_date': '2020-12-13T09:19:12'},
                 'p_59': {'duration': 26, '_date': '2020-12-13T09:19:38'},
                 'p_71': {'duration': 22, '_date': '2020-12-13T09:20:00'}}},
               'client': 'u_552873',
               'promotion_id': None,
               'payment_amount': 20.725,
               'discount_amount': 0,
               'basket': {'p_28': {'price': 3.0, 'category': 'p_c_7', 'rank': 1},
                'p_45': {'price': 9.375, 'category': 'p_c_9', 'rank': 92},
                'p_135': {'price': 0.255, 'category': 'p_c_1', 'rank': 188},
                'p_144': {'price': 4.095, 'category': 'p_c_6', 'rank': 70},
                'p_59': {'price': 2.32, 'category': 'p_c_1', 'rank': 15},
                'p_71': {'price': 1.68, 'category': 'p_c_3', 'rank': 155}},
               'total_products': 6,
               'session_start_date': '2020-12-13T09:16:06'}

            - At the end of each week, 'CHURN RATE' is also calculated and
              clients are randomly selected from the weekly_active_cliet list.
              These clients are removing from whole active clients not to get an order transaction from them.
        """
        self.connect_elastic_search()
        self.create_index()
        self.create_dates()
        self.create_customers()
        self.create_product_list()
        self.create_order_product_count()
        self.create_day_part_order_counts()
        self.create_promotions()
        self.get_actions()

        for d in self.days:
            orders = []
            daily_active_clients = []
            total_revenue = 0
            total_orders = 0
            week_part = 'week' if d[0].isoweekday() in range(6) else 'weekend'
            for day_part in ["nights", "mornings", "evenings"]:
                # orders
                order_count = random.sample(self.days_part_of_total_orders[week_part][day_part]["range"], 1)[0]
                active_clients = list(set(random.sample(self.client_active, order_count) +
                                          ['u_' + str(i) for i in random.sample(self._newcomer_list,
                                                                                int(len(self._newcomer_list) / 7))]))
                self.weekly_customers += active_clients
                hour_range = self.days_part_of_total_orders[week_part][day_part]["hours"]
                mins = list(range((hour_range[1] - hour_range[0]) * 60))
                date = d[0] + datetime.timedelta(hours=hour_range[0])
                print("date :", date, " ||day part :", hour_range[0])
                day_part_sessions = 0
                for o in active_clients:
                    _order_obj = {"id": None,
                                  "date": None,
                                  "actions": {},
                                  'client': None,
                                  "promotion_id": None,
                                  "payment_amount": None,
                                  "discount_amount": 0,
                                  "basket": {},
                                  "total_products": None,
                                  'session_start_date': None
                                  }
                    _has_promo = random.sample(self.has_promo, 1)[0]
                    _has_newcomer_promo = random.sample(self.has_newcomer_promo, 1)[0]
                    if o in self._newcomer_list:
                        if _has_newcomer_promo:
                            _order_obj['promotion_id'] = \
                            random.sample([_p for _p in self.promotions
                                           if self.promotions[_p]['category'] == 'newcomer'], 1)[0]
                    else:
                        if _has_promo:
                            _order_obj['promotion_id'] = \
                            random.sample([_p for _p in self.promotions
                                           if self.promotions[_p]['category'] != 'newcomer'], 1)[0]

                    _date = date + datetime.timedelta(minutes=random.sample(mins, 1)[0])
                    _order_obj['id'] = np.random.randint(200000000)
                    _order_obj['client'] = o
                    _order_obj['date'] = datetime.datetime.strptime(str(_date)[0:19], '%Y-%m-%d %H:%M:%S').isoformat()
                    _order_obj['total_products'] = random.sample(self.order_of_num_products, 1)[0]
                    _products = random.sample(self.products_list, _order_obj['total_products'])
                    _ranks = [self.products[p]['rank'] for p in _products]
                    _products = [p[1] for p in sorted(zip(_ranks, _products))]
                    _payment_amount = 0

                    # basket
                    _baskets = {}
                    _duration = 0
                    _actions = []
                    for p in _products:
                        _baskets[p] = {"price": self.products[p]['price'], "category": self.products[p]['category'],
                                       "rank": self.products[p]['rank']}
                        _payment_amount += self.products[p]['price']
                        _dur = self.order_basket_duration(p)
                        _actions.append({'product': p, 'duration': _dur})
                        _duration += _dur
                    _order_obj['basket'] = _baskets
                    _order_obj['payment_amount'] = _payment_amount
                    if _order_obj['promotion_id'] is not None:
                        _order_obj['discount_amount'] = _order_obj['payment_amount'] * \
                                                        self.promotions[_order_obj['promotion_id']]['discount_rate']
                    total_revenue += _payment_amount
                    total_orders += len(active_clients)
                    average_amount = total_revenue / total_orders if total_orders != 1 else 30
                    _duration += self.purchase_to_basket_duration(average_amount, _payment_amount)
                    _start_date = _date - datetime.timedelta(seconds=_duration)
                    _order_obj["session_start_date"] = datetime.datetime.strptime(str(_start_date)[0:19],
                                                                                  '%Y-%m-%d %H:%M:%S').isoformat()

                    # order actions
                    _order_obj["actions"]['add_to_basket'] = {}
                    for _d in _actions:
                        _start_date += datetime.timedelta(seconds=_d['duration'])
                        _order_obj["actions"]['add_to_basket'][_d['product']] = {'duration': _d['duration'],
                                                                                 '_date': datetime.datetime.strptime(
                                                                                     str(_start_date)[0:19],
                                                                                     '%Y-%m-%d %H:%M:%S').isoformat()
                                                                                 }

                    for a in ['has_sessions', 'has_basket', 'order_screen', 'purchased']:
                        _order_obj['actions'][a] = True
                    day_part_sessions += 1
                    # order has session - has basket before - order_screen before
                    has_sessions_before = random.sample(self.client_has_session_before_order_min, 1)[0]
                    if has_sessions_before != 0:
                        _ses_date = _start_date - datetime.timedelta(minutes=has_sessions_before)
                        _ses_obj = {"id": np.random.randint(200000000),
                                    "date": _ses_date.isoformat(),
                                    "actions": {'has_sessions': True,
                                                'has_basket': False,
                                                'order_screen': False,
                                                'purchased': False},
                                    'client': o,
                                    "promotion_id": None,
                                    "payment_amount": None,
                                    "discount_amount": 0,
                                    "basket": {},
                                    "total_products": None,
                                    'session_start_date': _ses_date.isoformat()
                                    }
                        day_part_sessions += 1
                        total_products = random.sample(self.order_of_num_products_for_no_orders, 1)[0]
                        if total_products != 0:
                            _ses_obj["actions"]["has_basket"] = True
                            _ses_obj['total_products'] = total_products
                            similar_products = []
                            similar_products_count = min(int(
                                random.sample(self.product_similarity_ratio_when_has_order_but_has_uncomplete_order, 1)[
                                    0] * len(_products)), len(_products))
                            if similar_products_count != 0:
                                similar_products = random.sample(_products, similar_products_count)
                            _products = random.sample(self.products_list,
                                                      len(_products) - len(similar_products)) + similar_products
                            # basket
                            _baskets = {}
                            _duration = 0
                            _actions = []
                            for p in _products:
                                _baskets[p] = {"price": self.products[p]['price'],
                                               "category": self.products[p]['category'],
                                               "rank": self.products[p]['rank']}
                                _payment_amount += self.products[p]['price']
                                _dur = self.order_basket_duration(p)
                                _actions.append({'product': p, 'duration': _dur})
                                _duration += _dur
                            _start_date = _ses_date - datetime.timedelta(seconds=_duration)
                            _ses_obj['basket'] = _baskets
                            _ses_obj['payment_amount'] = _payment_amount
                            _ses_obj['session_start_date'] = _start_date.isoformat()
                            # order actions
                            _ses_obj["actions"]['add_to_basket'] = {}
                            for _d in _actions:
                                _start_date += datetime.timedelta(seconds=_d['duration'])
                                _ses_obj["actions"]['add_to_basket'][_d['product']] = {'duration': _d['duration'],
                                                                                       '_date': datetime.datetime.strptime(
                                                                                           str(_start_date)[0:19],
                                                                                           '%Y-%m-%d %H:%M:%S').isoformat()
                                                                                       }

                            if random.sample(self.has_seen_basket_after_session, 1)[0]:
                                _ses_obj["actions"]["order_screen"] = True
                        helpers.bulk(self.es, self.get_insert_obj([_ses_obj], 'orders'))
                    daily_active_clients += active_clients
                    orders.append(_order_obj)
                # del _order_obj

                # sessions #########################
                session_count = int(random.sample(self.action_funnel_related_to_order['has_sessions'], 1)[
                                        0] * day_part_sessions) - day_part_sessions
                session_clients = random.sample(list(set(self.client_active) - set(daily_active_clients)), session_count)
                print("session client : ", len(session_clients))
                for s in session_clients:
                    _ses_date = date + datetime.timedelta(minutes=random.sample(mins, 1)[0])
                    _ses_obj = {"id": np.random.randint(200000000),
                                "date": _ses_date.isoformat(),
                                "actions": {'has_sessions': True,
                                            'purchased': False,
                                            'has_basket': False,
                                            'order_screen': False},
                                'client': s,
                                "promotion_id": None,
                                "payment_amount": None,
                                "discount_amount": 0,
                                "basket": {},
                                "total_products": None,
                                'session_start_date': _ses_date.isoformat()
                                }
                    total_products = random.sample(self.order_of_num_products_for_no_orders, 1)[0]
                    if total_products != 0:
                        _ses_obj["actions"]['has_basket'] = True
                        _ses_obj['total_products'] = total_products
                        _products = random.sample(self.products_list, _ses_obj['total_products'])
                        _ranks = [self.products[p]['rank'] for p in _products]
                        _products = [p[1] for p in sorted(zip(_ranks, _products))]
                        _payment_amount = 0
                        _baskets = {}
                        _duration = 0
                        _actions = []
                        # basket
                        for p in _products:
                            _baskets[p] = {"price": self.products[p]['price'],
                                           "category": self.products[p]['category'],
                                           "rank": self.products[p]['rank']}
                            _payment_amount += self.products[p]['price']
                            _dur = self.order_basket_duration(p)
                            _actions.append({'product': p, 'duration': _dur})
                            _duration += _dur
                        _ses_obj['basket'] = _baskets
                        _ses_obj['payment_amount'] = _payment_amount
                        _ses_obj['session_start_date'] = _ses_date - datetime.timedelta(seconds=_duration)

                        if random.sample(self.has_seen_basket_after_session, 1)[0]:
                            _ses_obj["actions"]["order_screen"] = True
                    helpers.bulk(self.es, self.get_insert_obj([_ses_obj], 'orders'))

            if d in self.weeks:
                _newcomer_list = self.customer_newcomer_list(random.sample(self.customer_newcomer_ratio, 1)[0],
                                                        len(self.weekly_customers))
                total_weekly_customers = len(set(self.weekly_customers))

                customer_orders = pd.DataFrame(self.weekly_customers).rename(columns={0: "client"})
                customer_orders['id'] = 1
                self.weekly_customers = list(
                    customer_orders.groupby("client").agg({"id": 'sum'}).reset_index().query("id == 1")[
                        'client'].unique())
                _churn_list = self.customer_churn_list(random.sample(self.customer_churn_ratio, 1)[0],
                                                       total_weekly_customers,
                                                  self.weekly_customers)
                print("week :", d)
                print("number of clients :", len(client_active))
                print("number of weekly_clients :", len(self.weekly_customers))
                print("churn clients :", len(_churn_list))
                print("newcomer clients :", len(_newcomer_list))
                client_active = list(set(client_active) - set(_churn_list)) + _newcomer_list
                print("updated num. of clients :", len(client_active))
                self.weekly_customers = []

            print(orders[0]['date'])
            helpers.bulk(self.es, self.get_insert_obj(orders, 'orders'))
            del orders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--host", type=str,
                        help="""
                                host for elasticsearch
                        """,
                        )
    parser.add_argument("-P", "--port", type=str,
                        help="""
                                port for elasticsearch
                        """,
                        )
    parser.add_argument("-SD", "--start_date", type=str,
                        help="""day of start for elasticsearch sample data

                        """,
                        )
    parser.add_argument("-ED", "--end_date", type=str,
                        help="""
                                day of end for elasticsearch sample data
                        """,
                        )
    parser.add_argument("-PDAY", "--prev_day_count", type=str,
                        help="""
                        previous number of date from the recent date in order to generate sample data
                        """)
    arguments = parser.parse_args()
    samples = CreateSampleIndex(host=arguments.host,
                                port=arguments.port,
                                start_date=arguments.start_date,
                                end_date=arguments.end_date,
                                prev_day_count=arguments.prev_day_count)

    samples.execute_sampling()
    samples.get_downloads()