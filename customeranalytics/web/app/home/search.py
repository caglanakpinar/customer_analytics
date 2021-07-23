import sys, os, inspect, logging
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from sqlalchemy import create_engine, MetaData
from os.path import join

try: from utils import convert_dt_to_day_str, abspath_for_sample_data, convert_to_date
except: from customeranalytics.utils import convert_dt_to_day_str, abspath_for_sample_data

try: from configs import query_path, default_es_port, default_es_host, default_message, schedule_columns
except: from customeranalytics.configs import query_path, default_es_port, default_es_host, default_message, schedule_columns


import pandas as pd
from datetime import datetime
from flask_login import current_user

try: from data_storage_configurations import connection_check, create_index, check_elasticsearch, QueryES
except: from customeranalytics.data_storage_configurations import connection_check, create_index, check_elasticsearch, QueryES

try: from exploratory_analysis import ea_configs
except: from customeranalytics.exploratory_analysis import ea_configs

try: from ml_process import ml_configs
except: from customeranalytics.ml_process import ml_configs

try: from web.app.home.forms import RealData
except: from customeranalytics.web.app.home.forms import RealData


engine = create_engine('sqlite://///' + join(abspath_for_sample_data(), "web", 'db.sqlite3'), convert_unicode=True,
                       connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


reports = RealData()


def ngrams_2(word):
    ngrams = []
    for i in range(len(word)):
        if word[i] != word[-1]:
            ngrams.append((word[i], word[i+1]))
    return ngrams


class Search:
    def __init__(self):
        """

        """
        self.temporary_path = None
        self.search_metrics = ['promotion', 'product', 'dimension']
        self.query_body = {"query": {}}
        self.intersect_count = lambda x, y: len(set(x) & set(y))
        self.user_data = pd.DataFrame()
        self.product_kpis = ['Average Product Sold Per Customer',
                             'Total Product Revenue',
                             'Total Product Discount', 'Number of Customer who purchase the Product']
        self.promotion_kpis = ['Average Daily Order with Promotion',
                               'Average Daily Revenue with Promotion',
                               'Average Daily Discount with Promotion',
                               'Unique Client Count who used the Promotion']
        self.product_charts = ["Total Number of Products per Day",
                               "Before - After Time Periods Customers' Total Purchase Count Test (Test Accepted!)",
                               "Before - After Time Periods Customers' Total Purchase Count Test (Test Rejected!)"]
        self.promotion_charts = ["Daily Order Count with Promotion",
                                 "Daily Revenue with Promotion",
                                 "Daily Discount with Promotion"]

    def get_temporary_path(self):
        try:
            es_con = pd.read_sql(""" SELECT *  FROM es_connection """, con).to_dict('results')[0]
            self.temporary_path = es_con['directory']
        except:
            connection, message = False, """
            ElasticSearch Connection Failed Check ES port/host or temporary path or Add new ElasticSearch connection
            """
            self.temporary_path = None

    def get_search_similarity_score(self, search_value, search_products):
        if self.intersect_count(search_value, search_products) != 0:
            return search_value, 1
        else:
            sv_ngram = ngrams_2(search_value)
            ngrams_intersections = []
            for i in search_products:
                _ngram = ngrams_2(i)
                ngrams_intersections.append((self.intersect_count(_ngram, sv_ngram), i))
            search_value = list(sorted(ngrams_intersections))[0]
            score = search_value[0] / len(sv_ngram)
            return search_value[1], score

    def get_serach_value(self, type, search_value):
        search_result, similarity_score = search_value, 0
        try:
            if type == 'product':
                products = list(self.collect_report('product_kpis').query("products == @search_value")['products'].unique())
                search_result, similarity_score = self.get_search_similarity_score(search_value, products)
            if type == 'promotion':
                promotions = self.collect_report('promotion_kpis')
                promotions = list(promotions.query("promotion_id == @search_value")['promotion_id'].unique())
                search_result, similarity_score = self.get_search_similarity_score(search_value, promotions)
            if type == 'client':
                clients = self.collect_report('client_kpis')
                clients = list(clients.query("clients == @search_value")['clients'].unique())
                search_result, similarity_score = self.get_search_similarity_score(search_value, clients)
        except Exception as e:
            search_result = search_value
        print(search_result, similarity_score)
        return search_result, similarity_score

    def collect_report(self, report_name):
        """
        If there is a report need as .csv format.
        """
        report = reports.fetch_report(report_name)
        if report is False:
            print("reports is not created")
            return pd.DataFrame()
        else:
            return report

    def visualization_data_for_product_search(self, value):
        for data_type in [('chart_1', ['product_kpis']),
                          ('chart_2', ['daily_products_of_sales']),
                          ('chart_3', ['product_usage_before_after_amount_accept',
                                       'product_usage_before_after_amount_reject']),
                          ('chart_4', ['product_usage_before_after_orders_accept',
                                       'product_usage_before_after_orders_reject']),
                          ]:
            try:
                result = pd.DataFrame()
                for r in data_type[1]:
                    result = pd.concat([result, self.collect_report(r).query("products == @value")])
                result.to_csv(join(self.temporary_path, "build_in_reports", "main", data_type[0] + '_search.csv'),
                              index=False)
            except Exception as e:
                print(e)

    def visualization_data_for_promotion_search(self, value):
        for data_type in [('chart_1', ['promotion_kpis']),
                          ('chart_2', ['daily_inorganic_ratio']),
                          ('chart_3', ['daily_promotion_revenue']),
                          ('chart_4', ['daily_promotion_discount']),
                          ]:
            try:
                result = pd.DataFrame()
                for r in data_type[1]:
                    _data = self.collect_report(r)
                    result = pd.concat([result, self.collect_report(r).query("promotion_id == @value")])
                if 'daily' in list(result.columns):
                    result['daily'] = result['daily'].apply(lambda x: convert_to_date(x))
                    result = result.sort_values('daily', ascending=True)
                result.to_csv(join(self.temporary_path, "build_in_reports", "main", data_type[0] + '_search.csv'),
                              index=False)
            except Exception as e:
                print(e)

    def visualization_data_for_client_search(self):
        for data_type in [('chart_1', ['client_kpis']),
                          ('chart_2', ['clv_predicted']),
                          ]:
            try:
                result = pd.DataFrame()
                for r in data_type[1]:
                    _data = self.collect_report(r)
                    result = pd.concat([result, self.collect_report(r).query("client == @value")])
                if 'daily' in list(result.columns):
                    result['daily'] = result['daily'].apply(lambda x: convert_to_date(x))
                    result = result.sort_values('daily', ascending=True)
                result.to_csv(join(self.temporary_path, "build_in_reports", "main", data_type[0] + '_search.csv'),
                              index=False)
            except Exception as e:
                print(e)


    def search_results(self, search_value):
        self.get_temporary_path()
        results = {'search_type': 'product', 'has_results': False, 'search_value': search_value}
        try:
            data = []
            for m in self.search_metrics:
                search_value, similarity_score = self.get_serach_value(m, search_value)
                data.append({'search_value': search_value, 'similarity_score': similarity_score, 'search_type': m})
            data = pd.DataFrame(data).query("similarity_score != 0")
            if len(data) != 0:
                data = data.sort_values(by='similarity_score', ascending=False).to_dict('results')[0]
                if data['search_type'] == 'product':
                    self.visualization_data_for_product_search(data['search_value'])
                if data['search_type'] == 'promotion':
                    self.visualization_data_for_promotion_search(data['search_value'])
                if data['search_type'] == 'client':
                    self.visualization_data_for_client_search(data['search_value'])
                data['has_results'] = True
                results = data
        except Exception as e:
            print(e)
            results = {'search_type': 'product', 'has_results': False}
        return results

    def get_search_chart_names(self, search_type):
        chart_names = {}
        if search_type == 'product':
            _charts = self.product_charts
            _kpis = self.product_kpis
        if search_type == 'promotion':
            _charts = self.promotion_charts
            _kpis = self.promotion_kpis
        for i in zip(range(2, 5), _charts):
            chart_names["chart_{}_search".format(str(i[0]))] = i[1]
        for i in zip(range(1, 5), _kpis):
            chart_names['kpi_' + str(i[0])] = i[1]
        return chart_names

    def convert_kpi_names_to_numeric_names(self, graph_json):
        return {'kpi_' + str(k[1]): graph_json['kpis'][k[0]] for k in zip(list(graph_json['kpis'].keys()), range(1, 5))}

    def delete_search_data(self, results):
        if results['has_results']:
            for i in range(1, 5):
                _file = join(self.temporary_path, "build_in_reports", "main", 'chart_{}_search.csv'.format(str(i)))
                try:
                    os.unlink(_file)
                except Exception as e:
                    print("no file is observed!!!")








