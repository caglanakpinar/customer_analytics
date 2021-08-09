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

from customeranalytics.configs import descriptive_reports, product_analytics, abtest_reports, non_dimensional_reports, \
    clv_prediction_reports, promotion_analytics
from customeranalytics.utils import *
from customeranalytics.data_storage_configurations.query_es import QueryES


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
        index : main || report :  daily_clv
        index : main || report :  daily_funnel
        index : main || report :  daily_funnel_downloads
        index : main || report :  weekly_average_payment_amount
        index : main || report :  product_usage_before_after_orders_reject
        index : main || report :  recency_clusters
        index : main || report :  weekly_cohort_from_2_to_3
        index : main || report :  daily_cohort_from_3_to_4
        index : main || report :  promotion_comparison
        index : main || report :  promotion_usage_before_after_orders_reject
        index : main || report :  most_ordered_products
        index : main || report :  daily_organic_orders
        index : main || report :  monetary_clusters
        index : main || report :  frequency_recency
        index : main || report :  segments_change_monthly_before_after_amount
        index : main || report :  daily_promotion_revenue
        index : main || report :  daily_cohort_from_2_to_3
        index : main || report :  weekly_cohort_from_3_to_4
        index : main || report :  product_usage_before_after_orders_accept
        index : main || report :  frequency_clusters
        index : main || report :  monthly_orders
        index : main || report :  daily_products_of_sales
        index : main || report :  hourly_funnel
        index : main || report :  weekly_average_order_per_user
        index : main || report :  daily_inorganic_ratio
        index : main || report :  daily_cohort_downloads
        index : main || report :  hourly_inorganic_ratio
        index : main || report :  promotion_number_of_customer
        index : main || report :  inorganic_orders_per_promotion_per_day
        index : main || report :  promotion_usage_before_after_orders_accept
        index : main || report :  recency_monetary
        index : main || report :  segments_change_monthly_before_after_orders
        index : main || report :  customer_journey
        index : main || report :  purchase_amount_distribution
        index : main || report :  dfunnel_anomaly
        index : main || report :  dcohort_anomaly_2
        index : main || report :  segments_change_weekly_before_after_orders
        index : main || report :  user_counts_per_order_seq
        index : main || report :  daily_cohort_from_1_to_2
        index : main || report :  hourly_organic_orders
        index : main || report :  weekly_cohort_downloads
        index : main || report :  hourly_funnel_downloads
        index : main || report :  product_usage_before_after_amount_accept
        index : main || report :  client_kpis
        index : main || report :  dorders_anomaly
        index : main || report :  churn
        index : main || report :  monthly_funnel_downloads
        index : main || report :  most_combined_products
        index : main || report :  avg_order_count_per_promo_per_cust
        index : main || report :  dcohort_anomaly
        index : main || report :  weekly_cohort_from_1_to_2
        index : main || report :  segments_change_daily_before_after_orders
        index : main || report :  rfm
        index : main || report :  promotion_usage_before_after_amount_accept
        index : main || report :  monthly_funnel
        index : main || report :  kpis
        index : main || report :  churn_weekly
        index : main || report :  clvsegments_amount
        index : main || report :  order_and_payment_amount_differences
        index : main || report :  hourly_orders
        index : main || report :  clvrfm_anomaly
        index : main || report :  weekly_funnel_downloads
        index : main || report :  product_usage_before_after_amount_reject
        index : main || report :  weekly_average_session_per_user
        index : main || report :  daily_promotion_discount
        index : main || report :  product_kpis
        index : main || report :  segments_change_weekly_before_after_amount
        index : main || report :  daily_orders
        index : main || report :  weekly_orders
        index : main || report :  segmentation
        index : main || report :  segments_change_daily_before_after_amount
        index : main || report :  most_ordered_categories
        index : main || report :  promotion_usage_before_after_amount_reject
        index : main || report :  monetary_frequency
        index : main || report :  promotion_kpis
        index : main || report :  client_feature_predicted

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
        self.promotion_analytics = promotion_analytics
        self.abtest_reports = abtest_reports
        self.clv_prediction_reports = clv_prediction_reports
        self.unexpected_reports = ["chart_{}_search".format(str(i)) for i in range(1, 5)]
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
        self.index_kpis = ['total_orders', 'total_visitors', 'total_revenue', 'total_discount',
                           'since_last_week_orders', 'since_last_week_revenue',
                           'since_last_week_total_visitors', 'since_last_week_total_discount']
        self.p_usage_ba_orders = ['promotion_usage_before_after_orders', 'promotion_usage_before_after_amount']
        self.rfm_reports = ['rfm', 'segmentation']
        self.clv_reports = ["clv_prediction", "stats", "segmentation"]
        self.rfm_metrics = {'recency', 'monetary', 'frequency'}
        self.day_folder = str(current_date_to_day())[0:10]
        self.rfm_metrics_reports = ['frequency_recency', 'recency_monetary', 'monetary_frequency',
                                    'monetary_clusters', 'frequency_clusters', 'recency_clusters']
        self.naming_decision = lambda x: 'no change' if x.split(" ")[1] == 'decrease/increase' else x.split(" ")[1]
        self.naming = lambda x1, x2: "Frequency; {0}, Monetary ; {1}".format(self.naming_decision(x1),
                                                                             self.naming_decision(x2))

    def connections(self):
        """
        connections to sqlite db. tables are;
            -   schedule_data
            -   es_connection
            -   data_columns_integration
        """
        tag, has_dimension = {}, False
        try:
            self.es_tag = pd.read_sql("SELECT * FROM es_connection", con).to_dict('results')[-1]
            dimensions = pd.read_sql("SELECT  * FROM data_connection", con).to_dict('results')[0]
            if dimensions['dimension'] not in ['None', None]:
                has_dimension = True
        except Exception as e:
           print(e)
        return has_dimension

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
                return ['main'] + dimensions
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
                            'partition': r['_source']['report_types'].get('partition', None),
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
        if r_name in self.promotion_analytics:
            query = " report_name == 'promotion_analytic' and type == '{}'".format(r_name)
        if r_name in 'segmentation':
            query = " report_name == 'segmentation'"
        if r_name in 'customer_journey':
            query = " report_name == 'cohort' and type == 'customers_journey'"
        if r_name in 'kpis':
            query = " report_name == 'stats' and type == ''"
        if r_name in 'rfm' or len(set(_splits) & self.rfm_metrics) != 0:
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
        if r_name in self.clv_prediction_reports:
            query = "report_name in ( 'clv_prediction', "
            if r_name == 'daily_clv':
                query += " 'stats') and type != type or type == 'daily_revenue' "
            if r_name == 'clvsegments_amount':
                query += " 'segmentation') and type != type "
        if 'anomaly' in r_name.split("_"):
            if r_name in ['dcohort_anomaly', 'dcohort_anomaly_2']:
                _type = 'cohort_d'
            if r_name == 'dfunnel_anomaly':
                _type = 'daily_funnel'
            if r_name == 'dorders_anomaly':
                _type = 'daily_orders_comparison'
            if r_name == 'clvrfm_anomaly':
                _type = 'clv_prediction'
            query = " report_name == 'anomaly' and type == '{}' ".format(_type)
        if 'churn' in r_name.split("_"):
            _type = 'overall'
            _time_period = list({'weekly', 'monthly'} & set(r_name.split("_")))
            _type = _time_period[0] if len(_time_period) != 0 else _type
            query = " report_name == 'churn' and type == '{}' ".format(_type)
        if r_name == 'client_kpis':
            query = "report_name in ('stats', 'rfm')"
        if r_name == 'client_feature_predicted':
            query = "report_name == 'clv_prediction' "
        if r_name == 'dimension_kpis':
            query = "report_name == 'stats' and type == 'dimension_kpis'"
        if r_name == 'daily_dimension_values':
            query = "report_name == 'stats' and type == 'daily_dimension_values'"
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
                self.p_usage_ba_orders[0]))['data'])[0]))
        usage_amount = self.required_aggregation(
            r_name=self.p_usage_ba_orders[1], data=pd.DataFrame(list(reports.query("abtest_type == '{}'".format(
                self.p_usage_ba_orders[1]))['data'])[0]))
        return pd.merge(usage_orders,
                        usage_amount.rename(columns={'diff': 'diff_amount'}),
                        on='promotions', how='inner')[['diff_amount', 'diff', 'promotions']]

    def get_clv_report(self, reports, r_name, index):
        clv_reports = reports.query("report_name == '{}' and type != type".format(self.clv_reports[0]))
        clv_partitions = list(set(clv_reports['partition'].unique()) - {None})
        if len(clv_partitions) == 0:
            clv = pd.DataFrame(list(reports.query("report_name == '{}' and type != type".format(
                                    self.clv_reports[0]))['data'])[0])
        else:
            clv = pd.DataFrame()
            for i in clv_partitions:
                clv = pd.concat([clv, pd.DataFrame(list(clv_reports.query("partition == @i")['data'])[0])])
        if index != 'main':
            clv = clv.query("dimension == @index")
        if r_name == 'daily_clv':
            daily_revenue = pd.DataFrame(list(reports.query("report_name == '{}' and type == 'daily_revenue'".format(
                    self.clv_reports[1]))['data'])[0]).rename(columns={"daily": "date"})
            clv['data_type'] = 'prediction'
            daily_revenue['data_type'] = "actual"
            reports = pd.concat([clv, daily_revenue])[['date', 'payment_amount', 'data_type']]
            reports = reports.groupby(['date', 'data_type']).agg({'payment_amount': 'sum'}).reset_index()
        if r_name == 'clvsegments_amount':
            segments = pd.DataFrame(list(reports.query("report_name == '{}'".format(self.clv_reports[2]))['data'])[0])
            reports = pd.merge(clv, segments, on='client', how='left').groupby("segments").agg(
                {"payment_amount": "sum"}).reset_index()
        return reports

    def radomly_sample_data(self, data):
        if len(data) > 500:
            data = data.sample(n=500, replace=False)
        return data

    def get_rfm_reports(self, reports, metrics=[]):
        rfm = pd.DataFrame(list(reports.query("report_name == '{}'".format(self.rfm_reports[0]))['data'])[0])
        segmentation = pd.DataFrame(list(reports.query("report_name == '{}'".format(self.rfm_reports[1]))['data'])[0])
        report = pd.merge(rfm, segmentation, on='client', how='left')
        if len(metrics) != 0:
            if 'clusters' not in metrics:
                report = report[metrics + ['segments_numeric']]
            else:
                _metric = list(set(metrics) & self.rfm_metrics)[0]
                report = report.groupby([_metric + '_segment', _metric]).agg(
                    {"client": lambda x: len(np.unique(x))}).reset_index().rename(columns={"client": "client_count"})
        return report

    def naming(self, metric):
        if metric == 'nomal decrease/increase':
            return 'no change'
        if metric == 'significant decrease':
            return 'decrease'
        if metric == 'significant increase':
            return 'increase'

    def get_anomaly_reports(self, r_name, report):
        report_data = pd.DataFrame(list(report['data'])[0])
        if r_name == 'dfunnel_anomaly':
            min_as = round(min(report_data['anomaly_scores']) - 0.01, 2)
            max_as = round(max(report_data['anomaly_scores']) + 0.01, 2)
            report_data['outlier'] = report_data['outlier'].apply(lambda x: max_as + 0.01 if x == 1 else min_as - 0.01)
            report_data = report_data.rename(columns={"anomaly_scores":"Anomaly Score Download to First Order"})
        if r_name == 'dcohort_anomaly':
            report_data['daily'] = report_data['daily'].apply(lambda x: convert_to_day(x))
            max_date = str(max(report_data['daily']) - datetime.timedelta(days=30))[0:10]
            report_data = report_data.query("daily > @max_date")
            outlier_days = list(report_data.query("outlier == 1")['daily'])
            _days = list(map(lambda x: str(x)[0:10] + '_outlier' if x in outlier_days else x, list(report_data['daily'])))
            report_data = report_data[[str(i) for i in list(range(0, 5))]].transpose().reset_index()
            report_data.columns = ['days'] + _days
        if r_name == 'dcohort_anomaly_2':
            report_data = report_data[['daily', 'anomaly_scores_from_d_to_1', 'outlier']
              ].rename(columns={"anomaly_scores_from_d_to_1": "Anomaly Score Download to First Order"})
        if r_name == 'dorders_anomaly':
            columns = ['diff_perc', 'daily', 'anomalities']
            report_data['anomalities'] = report_data['anomalities'].apply(lambda x: self.naming_decision(x))
            _normal = report_data.query("anomalities == 'no change'")[columns]
            _decrease = report_data.query("anomalities == 'decrease'")[columns]
            _increase = report_data.query("anomalities == 'increase'")[columns]
            report_data = pd.concat([_normal, _decrease, _increase])
        if r_name == 'clvrfm_anomaly':
            report_data['naming'] = report_data.apply(lambda row: self.naming(row['f_anomaly'], row['m_anomaly']), axis=1)
        return report_data

    def get_sample_report_names(self):
        """
        Collect all sample reports in sample_data folder
        """
        for f in listdir(dirname(self.folder)):
            if f.split(".")[1] == 'csv':
                _file = "_".join(f.split(".")[0].split("_")[2:])
                if _file not in self.unexpected_reports:
                    self.sample_report_names.append(_file)

    def calculate_last_week_ratios(self, k1, k2, k3):
        """

        """
        return k1 / (k2 - k3) if k2 - k3 != 0 else 0

    def get_last_week_kpis(self, report):
        """
        KPIs at index.html are stored at stats report. However there are reports
        which must be calculated calculated before store in built_in reports
        """
        report_data = pd.DataFrame(list(report['data'])[0])
        report_data['since_last_week_orders'] = report_data.apply(
            lambda row: self.calculate_last_week_ratios(row['total_orders'], row['last_week_orders'], row['last_week_revenue']), axis=1)
        report_data['since_last_week_revenue'] = report_data.apply(
            lambda row: self.calculate_last_week_ratios(row['last_week_revenue'], row['total_revenue'], row['last_week_revenue']), axis=1)
        report_data['since_last_week_total_visitors'] = report_data.apply(
            lambda row: self.calculate_last_week_ratios(row['last_week_visitors'], row['total_visitors'], row['last_week_visitors']), axis=1)
        report_data['since_last_week_total_discount'] = report_data.apply(
            lambda row: self.calculate_last_week_ratios(row['last_week_discount'], row['total_discount'], row['last_week_discount']), axis=1)
        return report_data[self.index_kpis]

    def get_client_kpis(self, report):
        """

        """
        total_order_count_per_customer = pd.DataFrame(list(report.query(
            "report_name == 'stats' and type == 'total_order_count_per_customer'").sort_values(
            'report_date', ascending=False)['data'])[0])
        rfm = pd.DataFrame(list(report.query(
            "report_name == 'rfm'").sort_values(
            'report_date', ascending=False)['data'])[0])[['frequency', 'recency', 'monetary', 'client']]
        report_data = pd.merge(total_order_count_per_customer, rfm, on='client', how='inner')
        return report_data

    def get_feature_predicted_data_per_customer(self, report):
        report_data = pd.DataFrame(list(report['data'])[0])
        return report_data.query("client != 'newcomers'")

    def get_related_report(self, reports, r_name, index):
        """
        Last exit before import data as .csv format
        """
        report_data = pd.DataFrame()
        report = reports.query(self.split_report_name(r_name))
        if len(report) != 0:
            if r_name == 'order_and_payment_amount_differences':
                report_data = self.get_order_and_payment_amount_differences(report)
            if r_name == 'rfm':
                report_data = self.get_rfm_reports(report)
            if r_name in self.rfm_metrics_reports:  # rfm rec. - mon.,
                report_data = self.get_rfm_reports(report, metrics=r_name.split("_"))
            if r_name not in ['order_and_payment_amount_differences', 'rfm', 'daily_clv'] + self.rfm_metrics_reports:
                report_data = self.required_aggregation(r_name, pd.DataFrame(list(report['data'])[0]))
            if r_name in ['daily_clv', "clvsegments_amount"]:
                report_data = self.get_clv_report(reports, r_name, index)
            if 'anomaly' in r_name.split("_"):
                report_data = self.get_anomaly_reports(r_name, report)
            if r_name in self.product_analytics + self.promotion_analytics:
                report_data = pd.DataFrame(pd.DataFrame(list(report['data'])[0]))
            if r_name == 'kpis':
                report_data = self.get_last_week_kpis(report)
            if r_name == 'client_kpis':
                report_data = self.get_client_kpis(report)
            if r_name == 'client_feature_predicted':
                report_data = self.get_feature_predicted_data_per_customer(report)
            if r_name in ['dimension_kpis', 'daily_dimension_values']:
                report_data = pd.DataFrame(list(report['data'])[0])

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

    def get_import_file_path(self, r_name, index, day_folder=False):
        """
        importing file path for .csv file to the build_in_reports
        """
        if not day_folder:
            return join(self.es_tag['directory'], "build_in_reports", index, r_name) + ".csv"
        else:
            return join(self.es_tag['directory'], "build_in_reports", index, self.day_folder, r_name) + ".csv"

    def collect_non_dimensional_reports(self):
        additional_reports = []
        for r in non_dimensional_reports:
            additional_reports.append(self.collect_reports(port=self.es_tag['port'],
                                                           host=self.es_tag['host'],
                                                           index='main',
                                                           query={"report_name": r}))
        return additional_reports

    def check_for_folder(self):
        try:
            os.mkdir(join(self.es_tag['directory'], "build_in_reports"))
        except:
            print("folder already exists")

    def check_for_index_folder(self, index):
        try: os.mkdir(join(self.es_tag['directory'], "build_in_reports", index))
        except: print("folder already exists")

    def check_for_day_folder(self, index):
        try: os.mkdir(join(self.es_tag['directory'], "build_in_reports", index, self.day_folder))
        except: print("folder already exists")

    def create_build_in_reports(self):
        """
        This is the main process of importing data into the given directory in .csv format.
        For each index, separate reports are imported.
        """
        self.get_sample_report_names()
        has_dimensions = self.connections()
        self.check_for_folder()

        if len(self.get_report_count(self.es_tag)) != 0:
            dimensions = self.collect_dimensions_for_data_works(has_dimensions)

            for index in dimensions:
                reports = self.collect_reports(self.es_tag['port'], self.es_tag['host'], index)
                reports = pd.concat([reports] + self.collect_non_dimensional_reports())
                reports['report_date'] = reports['report_date'].apply(lambda x: convert_to_day(x))
                reports = reports.sort_values(['report_name', 'report_date'],  ascending=False)
                if len(reports) != 0:  # if there has NOT been created any report, yet
                    try:
                        self.check_for_index_folder(index)
                        self.check_for_day_folder(index)
                        for r_name in self.sample_report_names:
                            print("index :", index, "|| report : ", r_name)
                            _data = self.get_related_report(reports, r_name, index)
                            if len(_data) != 0:
                                _data.to_csv(self.get_import_file_path(r_name, index), index=False)
                                _data.to_csv(self.get_import_file_path(r_name, index, day_folder=True), index=False)
                    except Exception as e:
                     pass

    def query_es_for_report(self, report_name, index, date=datetime.datetime.now()):
        ## TODO: will be updated
        """"
        """
        query = {'index': index,
                 'report_name': report_name,
                 'end': convert_to_iso_format(date),
                 'start': convert_to_iso_format(convert_to_day(date) - datetime.timedelta(days=1))}















