import sys, os, inspect
import pandas as pd
import numpy as np
from itertools import product

from ab_test_platform.executor import ABTest

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_es_port, default_es_host, default_query_date, time_periods
from customeranalytics.utils import *
from customeranalytics.data_storage_configurations.query_es import QueryES
from customeranalytics.ml_process.customer_segmentation import CustomerSegmentation


class ABTests:
    """
    AB Test;
    For more information pls check; https://pypi.org/project/abtest/


    """
    def __init__(self,
                 temporary_export_path,
                 has_product_connection=True,
                 has_promotion_connection=True,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        ******* ******** *****
        Dimensional AB Test:
        AB Tests must be created individually for dimensions.
        For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimension can be assigned in order to create the AB Test.

        download_index; downloads_location1 this will be the location dimension of
                        parameters in order to query downloads indexes; 'location1'.
        download_index; orders_location1 this will be the location dimension of
                        parameters in order to query orders indexes; 'location1'.
        ******* ******** *****

                IF THERE IS NO PROMOTION CONNECTION ON ORDERS INDEX, IT IS NOT STARTED !!
                IF THERE IS NO PRODUCT CONNECTION ON ORDERS INDEX, IT IS NOT STARTED !!

        :param has_product_connection: Has order index data for products?
        :param has_promotion_connection: Has order index data for promotions?
        :param host: elasticsearch host
        :param port: elasticsearch port
        :param download_index: elasticsearch port
        :param order_index: elasticsearch port
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.download_index = download_index
        self.order_index = order_index
        self.has_product_connection = has_product_connection
        self.has_promotion_connection = has_promotion_connection
        self.test = None
        self.query_es = QueryES(port=port, host=host)
        self.cs = CustomerSegmentation(port=self.port, host=self.host, order_index=self.order_index,
                                       download_index=self.download_index)
        self.data = pd.DataFrame()
        self.products = pd.DataFrame()
        self.promotions = []
        self.u_products = []
        self.max_date = current_date_to_day()
        self.dates = {}
        self.results = {}
        self.time_periods = time_periods[1:]
        self.path = temporary_export_path
        self.features = ["orders", "amount"]
        self.fields = ["client", "session_start_date", "payment_amount", "id"]
        self.fields += ["promotion_id"] if has_promotion_connection else []
        self.fields_products = ["id", "client", "basket", "session_start_date", "payment_amount"]
        self.test_groups = ['segments', 'promotions', "products", "payment_amount"]
        self.confidence_level = [0.01, 0.05]
        self.boostrap_ratio = [0.1, 0.2, 0.3]
        self.decision = pd.DataFrame()
        self.promotion_combinations = []
        self.promotion_comparison = pd.DataFrame()

    def dimensional_query(self, boolean_query=None):
        if dimension_decision(self.order_index):
            if boolean_query is None:
                boolean_query = [{"term": {"dimension": self.order_index}}]
            else:
                boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def get_orders_data(self, end_date):
        """
        Purchased orders are collected from Orders Index.
        Session_start_date, client, payment_amount are needed for initializing CLv Prediciton.
        :param end_date: final date to start clv prediction
        """
        self.query_es = QueryES(port=self.port, host=self.host)
        self.query_es.date_queries_builder({"session_start_date": {"lt": end_date}})
        self.query_es.query_builder(fields=self.fields, boolean_queries=self.dimensional_query())
        self.data = pd.DataFrame(self.query_es.get_data_from_es())

    def dimensional_query(self, boolean_query=None):
        if dimension_decision(self.order_index):
            if boolean_query is None:
                boolean_query = [{"term": {"dimension": self.order_index}}]
            else:
                boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def get_products(self, end_date):
        """
        Products for purchased orders are collected for Product Type A/B Test.
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
        IF THERE IS NO PRODUCTS DATA IS AVAILABLE SKIP THIS PROCESS !!!
        :param end_date: last date of data set
        """
        if self.has_product_connection:
            self.query_es = QueryES(port=self.port, host=self.host)
            self.query_es.date_queries_builder({"session_start_date": {"lt": end_date}})
            self.query_es.query_builder(fields=None,
                                        boolean_queries=self.dimensional_query([{"term": {"actions.has_basket": True}}]),
                                        _source=True)
            self.products = self.query_es.get_data_from_es()
            self.products = pd.DataFrame([{col: r['_source'][col] for col in self.fields_products} for r in self.products])
            self.products = self.products.query('basket == basket')
            self.products['products'] = self.products['basket'].apply(lambda x: list(x.keys()) if x == x else None)
            self.products = self.products.query('products == products')
            # get prices
            self.products['price'] = self.products.apply(
                lambda row: [row['basket'][i]['price'] for i in row['products']], axis=1)
            # merge price and product id into the list
            self.products['products'] = self.products.apply(
                lambda row: list(zip([row['id']] * len(row['products']),
                                     [row['client']] * len(row['products']),
                                     [row['payment_amount']] * len(row['products']),
                                     [row['session_start_date']] * len(row['products']),
                                     row['products'],
                                     row['price'])), axis=1)
            # concatenate products columns and convert it to dataframe with columns produc_id and price
            self.products = pd.DataFrame(np.concatenate(list(self.products['products']))).rename(
                columns={0: "id", 1: "client", 2: "payment_amount", 3: "session_start_date", 4: "products", 5: "price"})
            # subtract price from payment amount.
            # This will works to see how product of addition affects the total basket of payment amount
            self.products['payment_amount'] = self.products['payment_amount'].apply(lambda x: float(x))
            self.products['price'] = self.products['price'].apply(lambda x: float(x))
            self.products['session_start_date'] = self.products['session_start_date'].apply(lambda x: convert_to_day(x))

    def get_customer_segments(self):
        """
        Collecting segments of customers from the reports index.
        :param date: given report date
        """
        self.data = pd.merge(self.data, self.cs.fetch()[['client', 'segments']], on='client', how='left')

    def get_time_period(self, transactions, date_column):
        """
        converting date column of  values into the time_periods (hourly weekly, monthly,..)
        :param transactions: total data (orders data with actions)
        :return: data set with time periods
        """
        for p in list(zip(self.time_periods,
                     [convert_dt_to_day_str, find_week_of_monday, convert_dt_to_month_str])):
            transactions[p[0]] = transactions[date_column].apply(lambda x: p[1](x))
        transactions[date_column] = transactions[date_column].apply(lambda x: convert_to_date(x))
        return transactions

    def assign_organic_orders(self):
        """
        fill null promotions to 'organic'
        Check for promotion data connection
        :return:
        """
        if self.has_promotion_connection:
            self.data['promotion_id'] = self.data['promotion_id'].fillna('organic')

    def get_unique_promotions(self):
        """
        list of unique promotions
        Check for promotion data connection
        """
        if self.has_promotion_connection:
            self.promotions = list(self.data.query("promotion_id != 'organic'")['promotion_id'].unique())

    def get_unique_products(self):
        """
        list of unique products
        Check for products data connection.
        """
        if self.has_product_connection:
            self.u_products = list(self.products['products'].unique())

    def generate_test_groups(self):
        """
        generating test groups of AB Test. These groups are not A and B assigning process,
         It is the sub groups of the A and B samples
        :return:
        """
        self.test_groups = [("segments", tp) for tp in self.time_periods]
        if self.has_product_connection:
            self.test_groups += [("products", None)]
        if self.has_promotion_connection:
            self.test_groups += [("promotions", None)]

    def get_max_order_date(self):
        """
        max order date from session_start_date
        """
        self.max_date = max(list(self.data['daily']))

    def before_after_day_decision(self, day):
        """
        Before - After Test of date part decision
        :param day: interval date which splits dates into 2 parts before and after.
        :return: datetime
        """
        if day <= self.max_date:
            return day
        else:
            return self.max_date - datetime.timedelta(days=1)

    def generate_before_after_dates(self, date):
        """
        There 3 types of time periods to calculate dates.
            days, weeks, months
        :param date: current date
        """
        date = self.before_after_day_decision(date)
        _day = convert_to_date(date)
        _week = _day - datetime.timedelta(days=7)
        _month = _day - datetime.timedelta(days=30)
        _prev_day, _prev_week = [tp - datetime.timedelta(days=7) for tp in [_day, _week]]
        _prev_month = _month - datetime.timedelta(days=30)
        self.dates = {"daily": {"before": _prev_day, "after": _day},
                      "weekly": {"before": _prev_week, "after": _week},
                      "monthly": {"before": _prev_month, "after": _month}}

    def get_aggregation_func_and_renaming(self, feature):
        """
        There are 2 types of testing features;
            1.  Number of orders per customers on A - B Groups
            2.  Average Payment Amount per customer on A - B Groups
        :param feature: amount or order_count
        :return: aggfunc for groupby function
        """
        if feature == 'orders':
            return {'id': 'count'}, {'id': 'orders'}
        if feature == 'amount':
            return {'payment_amount': 'mean'}, {'payment_amount': 'amount'}

    def users_promotion_usage(self):
        self.get_unique_promotions()
        get_promo_data = lambda x, p: x.query("promotion_id == @p")
        _first_promo_orders = self.data.query("promotion_id != 'organic'").groupby(["client", "promotion_id"]).agg(
            {"session_start_date": "min"}).reset_index().rename(columns={"session_start_date": "promotion_date"})
        _promos = self.data.groupby(["client", "session_start_date", "id"]).agg(
            {"payment_amount": "mean"}).reset_index()
        befores, afters = [], []
        for p in self.promotions:
            _data = pd.merge(_promos,
                             get_promo_data(_first_promo_orders, p),
                             on='client', how='left')
            _b = _data.query("session_start_date < promotion_date").drop(
                'promotion_date', axis=1).rename(columns={"promotion_id": "promotions"})
            _b_clients = list(_b['client'].unique())
            afters.append(_data.query(
                "session_start_date >= promotion_date and client in @_b_clients").drop('promotion_date', axis=1))
            befores.append(_b)

        return pd.concat(befores), pd.concat(afters).rename(columns={"promotion_id": "promotions"})

    def users_product_usage(self):
        """
        Calculating the data for product usage.
            1.  There is only payment amount for product usage AB Testing.
            2.  Each customer of transactions before their order that purchased with the related product,
                Each customer of transactions after their order that purchased with the related product,
            3.  Filter out products data for 'before' data-frame
            4.  Collect the customers who has transaction on 'before' data-frame ('before' clients)
            5.  Filter out products data for 'after' data-frame with 'before' clients
        :return: concatenate products pf data-frames
        """
        self.get_unique_products()
        get_product_data = lambda x, p: x.query("products == @p")
        _first_products_orders = self.products.groupby(["client", "products"]).agg(
            {"session_start_date": "min", "price": "first"}).reset_index().rename(
            columns={"session_start_date": "product_date"})
        _products = self.products.groupby(
            ["client", "session_start_date", "id"]).agg({"payment_amount": "mean"}).reset_index()
        befores, afters = [], []
        for p in self.u_products:  # iteratively collect data for each product of before and after
                _data = pd.merge(_products,
                                 get_product_data(_first_products_orders, p),
                                 on='client', how='left')
                _data = _data.query("products == products")
                _data['payment_amount'] = _data['payment_amount'] - _data['price']
                _data = _data.drop('price', axis=1)
                _b = _data.query("session_start_date < product_date").drop('product_date', axis=1)
                _b_clients = list(_b['client'].unique())
                if len(_b) != 0:
                    afters.append(_data.query(
                        "session_start_date >= product_date and client in @_b_clients").drop('product_date', axis=1))
                    befores.append(_b)
        return pd.concat(befores), pd.concat(afters)

    def execute_test_grouping(self, data, metric, feature=None):
        """
        This allows us to trigger grouping part.
        :param data: data-frame
        :param metric: amount or order_count
        :param feature: grouping column additional to client
        :return: data-frame
        """
        _agg, renaming = self.get_aggregation_func_and_renaming(metric)
        _groups = [] if feature is None else [feature]
        return data.groupby(["client"] + _groups).agg(_agg).reset_index().rename(columns=renaming)

    def create_groups(self, metric, periods=None, feature=None, data=None, before_after_test=True):
        """
        This process is the creation of AB Test of A nd B groups
        :param metric: amount of order_count
        :param periods: if there is periodic AB Test such as Daily, Monthly, Weekly
        :param feature: products or promotions
        :param data: if there is data to manipulate
        :param before_after_test: if is before after test
        :return: concatenate data-frame
        """
        if before_after_test:
            if periods in ['daily', 'weekly', 'monthly']:
                _before = self.data[(self.data[periods] >= self.dates[periods]['before']) &
                                    (self.data[periods] < self.dates[periods]['after'])]
                _after = self.data[self.data[periods] >= self.dates[periods]['after']]
            else:
                if feature == 'promotions':
                    _before, _after = self.users_promotion_usage()
                if feature == 'products':
                    _before, _after = self.users_product_usage()

            dfs = []
            for group in [(_before, 'before'), (_after, 'after')]:
                print("group *******", group[1])
                _df = self.execute_test_grouping(data=group[0], metric=metric, feature=feature)
                _df['groups'] = group[1]
                dfs.append(_df)
            return pd.concat(dfs)

        else:
            data = self.execute_test_grouping(data=data, metric=metric, feature=feature)
            data['groups'] = '-'
            return data

    def decision_of_test(self, feature, groups, time_periods, comparison=False):
        """
        Creating a column of decision that any increase on A to B

        :param feature:amount or order_count
        :param groups: products, promotions
        :param time_periods: days, weeks, months
        :param comparison: if only comparing for before and after
        """
        if groups in ['promotions', 'products', 'segments']:
            self.decision = self.decision.groupby(groups).agg({"accept_Ratio": "mean",
                                                               "mean_control": "mean",
                                                               "mean_validation": "mean"}
                                                              ).reset_index().sort_values('accept_Ratio',
                                                                                          ascending=False)
            column = 'is_' + feature + '_increased_per_' + groups
            if groups == 'segments':
                column = column + "_per_" + time_periods
        if groups is None or comparison:
            self.decision = pd.DataFrame([{
                 i: np.mean(self.decision[i]) for i in ['accept_Ratio', 'mean_control', 'mean_validation']
            }])
            if groups is None:
                self.decision['time_period'] = time_periods
                column = 'is_' + feature + '_increased_per_' + time_periods
            if comparison:
                self.decision['promotion_comparison'] = groups[0] + '_' + groups[1]
                column = 'promo_1st_vs_promo_2nd'

        self.decision[column] = self.decision.apply(
            lambda row: True if row['mean_validation'] > row['mean_control'] and
                                row['mean_validation'] > 0.5 else False, axis=1)

    def decision_of_test_promo_comparison(self, decision, groups):
        """
        Creating a column of decision that any increase on A to B (only for promotion comparison)
        :param decision: data-frame
        :param groups: (promotion_1, promotion_2)
        :return:
        """
        decision = pd.DataFrame([{
                 i: np.mean(decision[i]) for i in ['accept_Ratio', 'mean_control', 'mean_validation']
            }])
        decision['promotion_comparison'] = groups[0] + '_' + groups[1]
        decision['1st promo'], decision['2nd promo'] = groups[0], groups[1]
        column = 'promo_1st_vs_promo_2nd'

        decision[column] = decision.apply(
            lambda row: True if row['mean_validation'] > row['mean_control'] and
                                row['mean_validation'] > 0.5 else False, axis=1)
        decision['total_positive_effects'] = decision['promo_1st_vs_promo_2nd'].apply(
            lambda x: 1 if x in ['True', True] else 0)
        return decision

    def name_of_test(self, is_before_after, fetaure, group, time_period):
        """

        :param is_before_after: True/False
        :param fetaure: amount, order_count
        :param group: promotion, product
        :param time_period: day, week, month
        :return:
        """
        name = ''
        if group == 'promotions':
            name += 'promotion_usage_'
        if group == 'products':
            name += 'product_usage_'
        if group == 'segments':
            name += 'segments_change_'
        if time_period is not None:
            name += time_period + '_'
        if is_before_after:
            name += 'before_after_'
        name += fetaure
        return name

    def create_before_after_test(self, date):
        """
        BEFORE - AFTER TEST:
            collect data from before the event and test with after the event.

        IF THERE IS NO PROMOTION CONNECTION ON ORDERS INDEX, IT IS NOT STARTED !!
        IF THERE IS NO PRODUCT CONNECTION ON ORDERS INDEX, IT IS NOT STARTED !!

        :param date: recent date
        :return:
        """
        self.results['customer_before_after'] = {}
        for groups in self.test_groups:
            for f in self.features:
                ab = ABTest(data=self.create_groups(metric=f, feature=groups[0], periods=groups[1]),
                            test_groups='groups',
                            groups=groups[0],
                            feature=f,
                            exporting_data=False,
                            export_path=self.path,
                            confidence_level=self.confidence_level,
                            boostrap_sample_ratio=self.boostrap_ratio)
                ab.ab_test_init()
                self.decision = ab.get_results()
                self.decision_of_test(f, groups[0], groups[1])
                _name = self.name_of_test(is_before_after=True, fetaure=f, group=groups[0], time_period=groups[1])
                self.insert_into_reports_index(self.decision,
                                               abtest_type=_name,
                                               index=self.order_index)
                del ab

                # self.decision.to_csv(join(self.path, _name + ".csv"), index=False)

    def execute_promotion_comparison_test(self, p):
        """
        Comparing all combination of Promotions
        :param p: promotion id
        :return: data-frame
        """
        ab = ABTest(data=self.create_groups(data=self.data[self.data['promotion_id'].isin([p[0], p[1]])],
                                            metric='amount',
                                            feature='promotion_id', before_after_test=False),
                    test_groups='promotion_id',
                    feature='amount',
                    groups="groups",
                    exporting_data=False,
                    export_path=self.path,
                    confidence_level=self.confidence_level,
                    boostrap_sample_ratio=self.boostrap_ratio)
        ab.ab_test_init()
        return self.decision_of_test_promo_comparison(decision=ab.get_results(), groups=[p[0], p[1]])

    def create_promotion_comparison_test(self, date):
        """
        Comparing all combination of Promotions
        IF THERE IS NO PROMOTION CONNECTION ON ORDERS INDEX, IT IS NOT STARTED !!
        :param date: recent date
        :return:
        """
        if self.has_promotion_connection:
            self.get_unique_promotions()
            self.promotion_combinations = list(filter(lambda x: x[0] != x[1],
                                                      list(product(self.promotions, self.promotions))))

            for p in self.promotion_combinations:
                self.promotion_comparison = pd.concat([self.promotion_comparison,
                                                       self.execute_promotion_comparison_test(p)])
            self.insert_into_reports_index(self.promotion_comparison,
                                           abtest_type='promotion_comparison',
                                           index=self.order_index)
            del self.promotion_comparison
            self.promotion_comparison = None

    def insert_into_reports_index(self,
                                  abtest,
                                  abtest_type,
                                  index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": start_date or current date,
         "report_name": "abtest",
         "index": "main",
         "report_types": {"abtest_type":  promotion_comparison || segments_change_monthly_before_after_amount, etc},
         "data": abtest.fillna(0.0).to_dict("results") -  dataframe to list of dictionary
         }
         !!! null values are assigned to 0.

        :param abtest: data set, data frame
        :param start_date: data start date
        :param abtest_type: orders, downloads, customer_journeys
        :param index: dimensionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = [{"id": np.random.randint(200000000),
                        "report_date": convert_to_day(current_date_to_day()).isoformat(),
                        "report_name": "abtest",
                        "index": get_index_group(index),
                        "report_types": {"abtest_type": abtest_type},
                        "data": abtest.fillna(0.0).to_dict("results")}]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def build_in_tests(self, date):
        """
        execute AB Test for given orders index
        :param date: recent date for query data
        :return:
        """
        _date = str(current_date_to_day())[0:10] if date is None else date
        self.get_orders_data(end_date=_date)
        self.get_products(end_date=_date)
        self.data = self.get_time_period(self.data, "session_start_date")
        self.assign_organic_orders()
        self.get_max_order_date()
        self.get_customer_segments()
        self.generate_before_after_dates(convert_to_day(_date))
        self.generate_test_groups()
        self.create_before_after_test(date)
        self.create_promotion_comparison_test(date)

    def fetch(self, abtest_name, start_date=None):
        """
        Example of cohort_name;
        promotion_comparison
        segments_change_monthly_before_after_amount
        segments_change_monthly_before_after_orders
        segments_change_weekly_before_after_amount
        segments_change_weekly_before_after_orders
        segments_change_daily_before_after_amount
        segments_change_daily_before_after_orders
        product_usage_before_after_amount
        product_usage_before_after_orders
        promotion_usage_before_after_amount
        promotion_usage_before_after_orders

        Directly, these arguments are sent to elasticsearch reports index in order to fetch related reports.

        Example AB Test Result;

            	segments	accept_Ratio	mean_control	mean_validation	is_amount_increased_per_segments_per_monthly
           0	lost	    0.633333	    24.180795	    23.996087	    False
           1	at risk	    0.616667	    23.587849	    23.914558	    True
           2	new customers	0.433333	31.431178	    31.835731	    True
           3	champions	0.233333	    18.990791	    20.332927	    True
           4	potential loyalist	0.133333	40.590746	39.738574	    False
           5	can`t lose them	0.066667	24.448692	    16.543292	    False
           6	need attention	0.016667	30.248727	    26.593158	    False
           7	loyal customers	0.000000	28.847650	    29.915854	    True
           8	others	   0.000000	        33.230453	    38.482046	    True
           9	promising	0.000000	    42.453413	    24.186591	    False

        :param abtest_name: e.g. promotion_comparison
        :param start_date: directly sending start_date to report_date in reports index.
        :return: data data-frame
        """

        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_name": "abtest"}},
                           {"term": {"report_types.abtest_type": abtest_name}},
                           {"term": {"index": get_index_group(self.order_index)}}]

        if start_date is not None:
            date_queries = [{"range": {"report_date": {"gte": convert_to_iso_format(start_date)}}}]

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

