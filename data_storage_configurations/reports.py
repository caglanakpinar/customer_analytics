import numpy as np
import pandas as pd
from numpy import unique
import sys, os, inspect
from sqlalchemy import create_engine, MetaData
from os.path import dirname, join
from os import listdir

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from configs import descriptive_reports, product_analytics, abtest_reports
from utils import *
from data_storage_configurations.query_es import QueryES


engine = create_engine('sqlite://///' + join(abspath_for_sample_data(), "web", 'db.sqlite3'), convert_unicode=True,
                       connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


class Reports:
    """
    There are some overall values that need to check each day for businesses.
    These values are also crucial metrics for the dashboards.
    These reports are collecting from 'reports' index and storing in a temporary folder.
    This folder path is a required field when ElasticSearch connection is created from the user.
    This process will be triggered when data storage process will be scheduled.
    Daily scheduling will be generated latest reports of the data
    Structure of the folder;
        - temporary_folder
            - build_in_reports
                - main
                    - weekly_funnel.csv
                    - daily_funnel.csv
                    ....
                - dimension_1
                    - weekly_funnel.csv
                    - daily_funnel.csv
                    ....
                - dimension_2
                    - weekly_funnel.csv
                    - daily_funnel.csv
                    ....


    Here are the reports;
        index : main || report :  weekly_funnel
        index : main || report :  daily_funnel
        index : main || report :  daily_funnel_downloads
        index : main || report :  weekly_average_payment_amount
        index : main || report :  weekly_cohort_from_2_to_3
        index : main || report :  daily_cohort_from_3_to_4
        index : main || report :  promotion_comparison
        index : main || report :  promotion_usage_before_after_orders_reject
        index : main || report :  most_ordered_products
        index : main || report :  daily_cohort_from_2_to_3
        index : main || report :  weekly_cohort_from_3_to_4
        index : main || report :  monthly_orders
        index : main || report :  hourly_funnel
        index : main || report :  weekly_average_order_per_user
        index : main || report :  daily_cohort_downloads
        index : main || report :  promotion_usage_before_after_orders_accept
        index : main || report :  customer_journey
        index : main || report :  purchase_amount_distribution
        index : main || report :  user_counts_per_order_seq
        index : main || report :  daily_cohort_from_1_to_2
        index : main || report :  weekly_cohort_downloads
        index : main || report :  hourly_funnel_downloads
        index : main || report :  monthly_funnel_downloads
        index : main || report :  weekly_cohort_from_1_to_2
        index : main || report :  rfm
        index : main || report :  promotion_usage_before_after_amount_accept
        index : main || report :  monthly_funnel
        index : main || report :  kpis
        index : main || report :  order_and_payment_amount_differences
        index : main || report :  hourly_orders
        index : main || report :  weekly_funnel_downloads
        index : main || report :  weekly_average_session_per_user
        index : main || report :  daily_orders
        index : main || report :  weekly_orders
        index : main || report :  segmentation
        index : main || report :  most_ordered_categories
        index : main || report :  promotion_usage_before_after_amount_reject

    There are some reports which still needs manipulation even they have been applied for data manipulation;
        - promotion_comparison
        - order_and_payment_amount_differences
        - promotion_usage_before_after_amount_reject/ _accept
        - promotion_usage_before_after_orders_reject/ _accept

    """

    def __init__(self):
        self.es_tag = {}
        self.folder = join(abspath_for_sample_data(), "exploratory_analysis", 'sample_data', '')
        self.sample_report_names = []
        self.get_main_query = lambda x: " time_period == '{0}' and report_name == '{1}' and type == '{2}' ".format(x[0],
                                                                                                                   x[1],
                                                                                                                   x[2])
        self.descriptive_reports = descriptive_reports
        self.product_analytics = product_analytics
        self.abtest_reports = abtest_reports
        self.double_reports = {'promotion_usage_before_after_amount': ['promotion_usage_before_after_amount_accept',
                                                                       'promotion_usage_before_after_amount_reject',
                                                                       'order_and_payment_amount_differences'],
                               'promotion_usage_before_after_orders': ['promotion_usage_before_after_orders_accept',
                                                                       'promotion_usage_before_after_orders_reject'],
                               'product_usage_before_after_amount': ['product_usage_before_after_amount_accept',
                                                                     'product_usage_before_after_amount_reject'],
                               'product_usage_before_after_orders': ['product_usage_before_after_orders_accept',
                                                                     'product_usage_before_after_orders_reject']
                              }
        self.p_usage_ba_orders = ['promotion_usage_before_after_orders', 'promotion_usage_before_after_amount']
        self.rfm_reports = ['rfm', 'segmentation']
        self.day_folder = str(current_date_to_day())[0:10]

    def connections(self):
        """
        connections to sqlite db. tables are;
            -   schedule_data
            -   es_connection
            -   data_columns_integration
        """
        tag, has_dimension = {}, False
        try:
            tag = pd.read_sql("SELECT * FROM schedule_data", con).to_dict('resutls')[-1]
            self.es_tag = pd.read_sql("SELECT * FROM es_connection", con).to_dict('resutls')[-1]
            dimensions = pd.read_sql("SELECT  * FROM data_columns_integration", con).to_dict('results')[0]
            if dimensions['dimension'] not in ['None', None]:
                has_dimension = True
        except Exception as e:
            print(e)
        return tag, has_dimension

    def collect_dimensions_for_data_works(self, has_dimensions):
        """
        If there is dimension in the orders Index all reports will be created individually per indexes
        with 'main' which indicates whole data in orders index
        """
        dimensions = []
        if has_dimensions:
            try:
                qs = QueryES(host=self.es_tag['host'], port=self.es_tag['port'])
                _res = qs.es.search(index='orders', body={"size": 0,
                                                                "aggs": {"langs": {
                                                                         "terms": {"field": "dimension.keyword",
                                                                                   "size": 500}
                                                                         }}})
                _res = [r['key'] for r in _res['aggregations']['langs']['buckets']]
                dimensions = unique(_res).tolist()
            except Exception as e:
                print(e)
            if dimensions not in ['None', None] and len(dimensions) != 1:
                return dimensions + ['main']
            else:
                return ['main']
        else:
            return ['main']

    def collect_reports(self, port, host, index, query=None):
        """
        Query reports index with given date
        """
        query_es = QueryES(host=host, port=port)
        res = []

        match = {"size": 1000, "from": 0}
        if query is not None:
            date_queries = None if 'end' not in list(query.keys()) else [
                {'range': {'report_date': {'lt': query['end']}}}]

            boolean_queries = []
            _keys = list(query.keys())
            for _b in ['index', 'report_name']:
                if _b in _keys:
                    boolean_queries.append({'term': {_b: query[_b]}})
            try:
                boolean_queries = None if len(boolean_queries) == 0 else boolean_queries
                query_es.query_builder(boolean_queries=boolean_queries,
                                       date_queries=date_queries,
                                       fields=None,
                                       _source=True)
                match = query_es.match
            except Exception as e:
                print(e)

        for r in query_es.es.search(index='reports', body=match)['hits']['hits']:
            if r['_source']['index'] == index:
                try:
                    _obj = {'report_name': r['_source']['report_name'],
                            'report_date': r['_source']['report_date'],
                            'time_period': r['_source']['report_types'].get('time_period', None),
                            'type': r['_source']['report_types'].get('type', None),
                            '_from': r['_source']['report_types'].get('from', None),
                            '_to': r['_source']['report_types'].get('to', None),
                            'abtest_type': r['_source']['report_types'].get('abtest_type', None),
                            'report_types': r['_source']['report_types'],
                            'index': r['_source']['index'],
                            'data': r['_source']['data']
                            }
                    res.append(_obj)

                except Exception as e:
                    print(e)
        return pd.DataFrame(res)

    def split_report_name(self, r_name):
        """
        get related data from reports index for the given report name
        """
        _splits = r_name.split("_")
        query = ''
        if 'funnel' in _splits:
            query = self.get_main_query((_splits[0], _splits[1], _splits[2] if len(_splits) == 3 else 'orders'))

        if 'cohort' in _splits:
            query = self.get_main_query((_splits[0], _splits[1], 'downloads' if _splits[-1] == 'downloads' else 'orders'))
            if 'from' in _splits:
                query += " and _from == '{}' and _to == '{}' ".format(_splits[3], _splits[5])
        if r_name in self.descriptive_reports:
            query = " report_name == 'stats' and type == '{}'".format(r_name)
        if r_name in self.product_analytics:
            query = " report_name == 'product_analytic' and type == '{}'".format(r_name)
        if r_name in 'segmentation':
            query = " report_name == 'segmentation'"
        if r_name in 'customer_journey':
            query = " report_name == 'cohort' and type == 'customers_journey'"
        if r_name in 'kpis':
            query = " report_name == 'stats' and type == ''"
        if r_name in 'rfm' or len(set(_splits) & {'recency', 'monetary', 'frequency'}) != 0:
            query = " report_name in ('rfm', 'segmentation')"

        # ab-test reports
        if r_name == 'promotion_comparison':
            query = " report_name == 'abtest' and abtest_type == 'promotion_comparison'"
        if r_name == 'order_and_payment_amount_differences':
            query = " report_name == 'abtest' and abtest_type in ('{}')".format("', '".join(self.p_usage_ba_orders))
        if r_name in abtest_reports and r_name not in ['promotion_comparison', 'order_and_payment_amount_differences']:
            query = " report_name == 'abtest' and abtest_type == '{}'".format(r_name)
        if 'usage' in _splits:
            for sub_reports in self.double_reports:
                if r_name in self.double_reports[sub_reports]:
                    query = " report_name == 'abtest' and abtest_type == '{}'".format(sub_reports)

        if r_name == 'user_counts_per_order_seq':
            query = " report_name == 'stats' and type == 'user_counts_per_order_seq' "
        print(r_name, query)
        return query

    def get_promotion_comparison(self, x):
        """
        This is only for promotion_comparison reports. Promotion fields is stored in the index promo1_promo_2.
        """
        _x = x.split("_")
        return pd.Series(["_".join(_x[0:2]), "_".join(_x[2:4])])

    def required_aggregation(self, r_name, data):
        """
        These is the last data manipulation before stored in the directory.
        """
        if r_name == 'segmentation':
            total_clients = len(data)
            data = data.groupby("segments").agg({"client": "count"}).reset_index().rename(columns={"client": "value"})
            data['value'] = round((data['value'] * 100) / total_clients, 2)
        if r_name in self.abtest_reports:
            data['diff'] = data['mean_validation'] - data['mean_control']
            if 'usage' in r_name.split("_"):
                accept = 'True' if 'accept' in r_name.split("_") else 'False'
                metric = 'amount' if 'amount' in r_name.split("_") else 'orders'
                data = data.query("is_" + metric + "_increased_per_promotions == " + accept)
                data = data.sort_values('diff') if 'accept' in r_name.split("_") else data
            if r_name == 'promotion_comparison':
                data[['1st promo', '2nd Promo']] = data['promotion_comparison'].apply(
                    lambda x: self.get_promotion_comparison(x))
                data['total_effects'] = data['promo_1st_vs_promo_2nd'].apply(
                    lambda x: 1 if x in ['True', True] else 0)
                data = data.groupby("1st promo").agg({"total_effects": "sum",
                                                      "accept_Ratio": "mean",
                                                      "promo_1st_vs_promo_2nd": "count"}).reset_index()
                data = data.sort_values(['total_effects'], ascending=False).sort_values(['accept_Ratio'], ascending=True)
                data = data.rename(columns={"promo_1st_vs_promo_2nd": "total_negative_effects"})
                data['total_negative_effects'] = data['total_negative_effects'] - data['total_effects']
        if 'most' in r_name.split("_"):
            if r_name == 'most_combined_products':
                data = data.rename(columns={"product_pair": "products", "total_pairs": "order_count"})
            data = data.sort_values(by='order_count', ascending=False).iloc[:20].reset_index(drop=True)
        return data

    def get_order_and_payment_amount_differences(self, reports):
        """
        It is only for 'order_and_payment_amount_differences'. It is the combination of data
        which are 'promotion_usage_before_after_orders' and 'promotion_usage_before_after_amount'.
        """
        usage_orders = self.required_aggregation(
            r_name=self.p_usage_ba_orders[0], data=pd.DataFrame(list(reports.query("abtest_type == '{}'".format(
                self.p_usage_ba_orders[0])).sort_values('report_date',  ascending=False)['data'])[0]))
        usage_amount = self.required_aggregation(
            r_name=self.p_usage_ba_orders[1], data=pd.DataFrame(list(reports.query("abtest_type == '{}'".format(
                self.p_usage_ba_orders[1])).sort_values('report_date', ascending=False)['data'])[0]))
        return pd.merge(usage_orders,
                        usage_amount.rename(columns={'diff': 'diff_amount'}),
                        on='promotions', how='inner')[['diff_amount', 'diff', 'promotions']]

    def radomly_sample_data(self, data):
        if len(data) > 500:
            data = data.sample(n=500, replace=False)
        return data

    def get_rfm_reports(self, reports, metrics=[]):
        rfm = pd.DataFrame(list(reports.query("report_name == '{}'".format(
                self.rfm_reports[0])).sort_values('report_date',  ascending=False)['data'])[0])
        segmentation = pd.DataFrame(list(reports.query("report_name == '{}'".format(
                self.rfm_reports[1])).sort_values('report_date',  ascending=False)['data'])[0])
        report = pd.merge(rfm, segmentation, on='client', how='left')
        if len(metrics) != 0:
            report = report[metrics + ['segments_numeric']]
        return report

    def get_sample_report_names(self):
        """
        Collect all sample reports in sample_data folder
        """
        for f in listdir(dirname(self.folder)):
            if f.split(".")[1] == 'csv':
                self.sample_report_names.append("_".join(f.split(".")[0].split("_")[2:]))

    def get_related_report(self, reports, r_name):
        """
        Last exit before import data as .csv format
        """
        report_data = pd.DataFrame()
        report = reports.query(self.split_report_name(r_name))
        if len(report) != 0:
            report['report_date'] = report['report_date'].apply(lambda x: convert_to_day(x))
            if r_name == 'order_and_payment_amount_differences':
                report_data = self.get_order_and_payment_amount_differences(report)
            if r_name == 'rfm':
                report_data = self.get_rfm_reports(report)
            if len(set(r_name.split("_")) & {'recency', 'monetary', 'frequency'}) != 0:
                report_data = self.get_rfm_reports(report, metrics=r_name.split("_"))
            if r_name not in ['order_and_payment_amount_differences', 'rfm']:
                report = report.sort_values('report_date', ascending=False)
                report_data = self.required_aggregation(r_name, pd.DataFrame(list(report['data'])[0]))
        report_data = self.radomly_sample_data(report_data)
        return report_data

    def get_report_count(self, es_tag):
        """
        Check if the reports are stored in to the reports index.
        """
        reports_index_count = {}
        try:
            qs = QueryES(host=es_tag['host'], port=es_tag['port'])
            reports_index_count = qs.es.cat.count('reports', params={"format": "json"})
        except Exception as e:
            print(e)
        return reports_index_count

    def create_build_in_reports(self):
        """
        This is the main process of importing data into the given directory in .csv format.
        For each index, separate reports are imported.
        """
        self.get_sample_report_names()
        tag, has_dimensions = self.connections()
        try:
            os.mkdir(join(self.es_tag['directory'], "build_in_reports"))
        except:
            print("folder already exists")
        if len(self.get_report_count(self.es_tag)) != 0:
            dimensions = self.collect_dimensions_for_data_works(has_dimensions)
            for index in dimensions:
                reports = self.collect_reports(self.es_tag['port'], self.es_tag['host'], index)
                try:
                    os.mkdir(join(self.es_tag['directory'], "build_in_reports", index))
                except:
                    print("folder already exists")
                try:
                    os.mkdir(join(self.es_tag['directory'], "build_in_reports", index, self.day_folder))
                except:
                    print("folder already exists")
                for r_name in self.sample_report_names:
                    print("index :", index, "|| report : ", r_name)
                    _data = self.get_related_report(reports, r_name)
                    if len(_data) != 0:
                        _data.to_csv(join(self.es_tag['directory'], "build_in_reports", index, r_name) + ".csv",
                                     index=False)
                        _data.to_csv(
                            join(self.es_tag['directory'], "build_in_reports", index, self.day_folder, r_name) + ".csv",
                            index=False)

    def query_es_for_report(self, report_name, index, date=datetime.datetime.now()):
        ## TODO: will be updated
        """"
        """
        query = {'index': index,
                 'report_name': report_name,
                 'end': convert_to_iso_format(date),
                 'start': convert_to_iso_format(convert_to_day(date) - datetime.timedelta(days=1))}















