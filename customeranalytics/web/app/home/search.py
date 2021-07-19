import sys, os, inspect, logging
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from sqlalchemy import create_engine, MetaData
from os.path import abspath, join
from numpy import mean

try: from utils import convert_dt_to_day_str, abspath_for_sample_data
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
        self.search_metrics = ['promotion', 'product', 'dimension', 'client']
        self.query_body = {"query": {}}
        self.intersect_count = lambda x, y: len(set(x) & set(y))
        self.user_data = pd.DataFrame()

    def create_es_query_body(self, key, value):
        if key == 'product':
            self.query_body['query'] = {"bool": {"must": [{"term": {"actions.purchased": True}},
                                                          {"exists": {"field": "basket." + value}}]}}
        else:
            self.query_body['query'] = {"bool": {"must": [{"term": {"actions.purchased": True}},
                                                          {"prefix": {"client": value}}]}}

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
            score = list(sorted(ngrams_intersections))[1] / len(sv_ngram)
            return search_value, score

    def get_serach_value(self, type, search_value):
        self.create_es_query_body(type, search_value)
        search_result, similarity_score = search_value, 0
        try:
            if type == 'product':
                products = list(self.collect_report('product_kpis').query("products == @value")['products'].unique())
                search_result, similarity_score = self.get_search_similarity_score(search_value, products)

        except: search_result = search_value
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
                          ('chart_2', ['daily_products']),
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

    def search_results(self, search_value):
        results = {'search_type': 'product', 'has_results': False}
        try:
            data = []
            for m in self.search_metrics:
                search_value, similarity_score = self.get_serach_value(m, search_value)
                data.append({'search_value': search_value, 'similarity_score': similarity_score, 'search_type': m})
            data = pd.DataFrame(data).query("similarity_score != 0")
            if len(data) != 0:
                data = data.sort_values(by='similarity_score', ascending=False).to_dict('results')[0]
                if data['search_type'] == 'product':
                    self.visualization_data_for_product_search(results['similarity_score'])
                data['has_results'] = True
                results = data.to_dict('results')
        except: results = {'search_type': 'product', 'has_results': False}
        return results



