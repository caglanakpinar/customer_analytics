import sys, os, inspect, logging
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from sqlalchemy import create_engine, MetaData
from os.path import join

try: from utils import convert_dt_to_day_str, abspath_for_sample_data, convert_to_date
except: from customeranalytics.utils import convert_dt_to_day_str, abspath_for_sample_data

try: from configs import query_path, default_es_port, default_es_host, default_message, schedule_columns, data_types_for_search
except: from customeranalytics.configs import query_path, default_es_port, default_es_host, default_message, schedule_columns, data_types_for_search


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
    """
    word with string form is converted to 2 ngrams
    Example;
        'test'; (t, e), (e, s), (s, t)
    :param word:
    """
    ngrams = []
    for i in range(len(word)):
        if word[i] != word[-1]:
            ngrams.append((word[i], word[i+1]))
    return ngrams


class Search:
    def __init__(self):
        """
        There are 4 types of search;
            - product
            - client
            - promotion
            - dimension

        Each type of search represents an individual dashboard with results.
        When expected search value is typed in the search bar;
            1. Results are checked individually for each type of search.
                a.  create ngrams of search value
                b.  create ngrams for each list of search types.
                c.  calculate the similarity between ngrams of search value and ngrams for each list of search types.
                d.  remove score = 0.
                e.  find the top score and assign it as a detected search result.
            2. Create a temporary .csv file at temporary_folder_path that is assigned by the user.
                These are 3 charts of .csv files;
                 a. chart_2_search.csv is positioned at right top,
                 b. chart_3_search.csv is positioned at left bottom,
                 c. chart_4_search.csv is positioned at right bottom,
                There are 4 KPIs with 1 .csv file;
                 a. chart_1_search.csv is positioned ad the left top. These KPIs will be changed

            3. Each chart of data is created at temporary file
               (chart_1_search.csv, .., chart_4_search.csv)  will be removed after the dashboard is shown at the user interface.
            4. Each search type has individual charts and KPIs.
               So, each creation of charts and KPIs .csv files must be applied individually.

        """
        self.temporary_path = None
        self.search_metrics = ['promotion', 'product', 'client', 'dimension']
        self.query_body = {"query": {}}
        self.intersect_count = lambda x, y: len(set(x) & set(y))
        self.user_data = pd.DataFrame()
        # product, promotion, dimension, client search charts and KPIs
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
        self.client_charts = ["Possible Future Purchase of the Customer"]
        self.client_kpis = ["Total Order Count of the Customer",
                            "Order Frequency of the Customer (hr)",
                            "Total Hour Count from the last time the customer purchased (recency)",
                            "Average Payment Amount of the Customer"]
        self.dimension_charts = ["Daily Total Order Count With Dimension",
                                 "Daily Total Purchase Amount With Dimension",
                                 "Daily Total Unique Client Count With Dimension"]
        self.dimension_kpis = ["Total Order Count", "Total Purchase Amount",
                               "Total Unique Client Count", "Total Discount Amount"]

        self.data_types = data_types_for_search
        self.query_column = {'product': "products", "client": "client",
                             "promotion": "promotion_id", "dimension": "dimension"}

    def get_temporary_path(self):
        """
        query for fetching the temporary folder path
        """
        try:
            es_con = pd.read_sql(""" SELECT *  FROM es_connection """, con).to_dict('results')[0]
            self.temporary_path = es_con['directory']
        except:
            connection, message = False, """
            ElasticSearch Connection Failed Check ES port/host or temporary path or Add new ElasticSearch connection
            """
            self.temporary_path = None

    def get_search_similarity_score(self, search_value, search_list):
        """
        similarity score with ngrams 2
        similarity score =
                the total intersection of ngrams (searching word & word from the searching list) / count of ngrams of searching word

        :param search_value: value for searching at products, dimensions, clients, promotions
        :param search_list:  list of searching data
        """
        if self.intersect_count([search_value], search_list) > 0:
            return search_value, 1
        else:
            sv_ngram = ngrams_2(search_value)  # searching word of 2 ngrams
            ngrams_intersections = []
            for i in search_list:
                _ngram = ngrams_2(i)  # all possible search list of 2 ngrams
                ngrams_intersections.append((self.intersect_count(_ngram, sv_ngram), i))
            search_value = list(sorted(ngrams_intersections))[-1]
            score = search_value[0] / len(sv_ngram)
            return search_value[1], score

    def get_search_value(self, type, search_value):
        """
        :param type         : client, product, dimension, promotion
        :param search_value : value for searching at products, dimensions, clients, promotions
        """
        search_result, similarity_score, searches = search_value, 0, []
        try:
            if type == 'product':
                searches = list(self.collect_report('product_kpis')['products'].unique())
            if type == 'promotion':
                searches = list(self.collect_report('promotion_kpis')['promotion_id'].unique())
            if type == 'client':
                searches = list(self.collect_report('client_kpis')['client'].unique())
            if type == 'dimension':
                searches = list(self.collect_report('dimension_kpis')['dimension'].unique())
            search_result, similarity_score = self.get_search_similarity_score(search_value, searches)
        except Exception as e:
            print(e)
            search_result = search_value
        return search_result, similarity_score

    def collect_report(self, report_name):
        """
        If there is a report need as .csv format
        :param report_name: name of the report fetch from build-in temporary report folder
        """
        report = reports.fetch_report(report_name)
        if report is False:
            print("reports is not generated")
            return pd.DataFrame()
        else:
            return report

    def visualization_data_for_search(self, type, value):
        """
        Each type of search is visualized individually.
            Products; 4 KPIs and 3 charts
            Promotions; 4 KPIs and 3 charts
            Clients; 4 KPIs and 1 chart
            Dimensions; 4 KPIs and 3 charts

        Each chart and KPI of data fetch from imported .csv files at temporary report folder
        after built-in report processes are finalized.
        :param type: search type; promotions, products, clients, dimensions
        :param value: value for searching
        """
        _query = "{} == @value".format(self.query_column[type])
        for data_type in self.data_types[type]:
            try:
                result = pd.DataFrame()
                for r in data_type[1]:
                    result = pd.concat([result, self.collect_report(r).query(_query)])
                if len({'date', 'daily'} & set(list(result.columns))) != 0:  # covert date columns to timestamp
                    date_column = list({'date', 'daily'} & set(list(result.columns)))[0]
                    result[date_column] = result[date_column].apply(lambda x: convert_to_date(x))
                    result = result.sort_values(date_column, ascending=True)
                result.to_csv(join(self.temporary_path, "build_in_reports", "main", data_type[0] + '_search.csv'),
                              index=False)
            except Exception as e:
                print(e)

    def search_results(self, search_value):
        """
        1. check for the temporary folder path.
        2. check each search type separately and calculate the max similarity scores.
        3. if there is not observed score of more than 0, show the sample data on search dashboard.

        :param value: value for searching
        """
        self.get_temporary_path()
        results = {'search_type': 'product', 'has_results': False, 'search_value': search_value}
        try:
            data = []
            for m in self.search_metrics:
                _search_value, _similarity_score = self.get_search_value(m, search_value)
                data.append({'search_value': _search_value, 'similarity_score': _similarity_score, 'search_type': m})
            data = pd.DataFrame(data).query("similarity_score != 0")
            if len(data) != 0:
                data = data.sort_values(by='similarity_score', ascending=False).to_dict('results')[0]
                self.visualization_data_for_search(type=data['search_type'], value=data['search_value'])
                data['has_results'] = True
                results = data
        except Exception as e:
            results = {'search_type': 'product', 'has_results': False}
        return results

    def get_search_chart_names(self, search_type):
        """
        These are the title of the charts and KPIs at the search dashboard according to the search type.
        :param search_type: search type; promotions, products, clients, dimensions
        """
        chart_names = {}
        if search_type == 'product':
            _charts = self.product_charts
            _kpis = self.product_kpis
        if search_type == 'promotion':
            _charts = self.promotion_charts
            _kpis = self.promotion_kpis
        if search_type == 'client':
            _charts = self.client_charts
            _kpis = self.client_kpis
        if search_type == 'dimension':
            _charts = self.dimension_charts
            _kpis = self.dimension_kpis
        for i in zip(range(2, len(_charts) + 2), _charts):
            chart_names["chart_{}_search".format(str(i[0]))] = i[1]
        for i in zip(range(1, len(_kpis) + 1), _kpis):
            chart_names['kpi_' + str(i[0])] = i[1]
        return chart_names

    def convert_kpi_names_to_numeric_names(self, graph_json):
        """
        :param graph_json: json file for charts and KPIs in order to show on .html file
         """
        return {'kpi_' + str(k[1]): graph_json['kpis'][k[0]] for k in zip(list(graph_json['kpis'].keys()), range(1, 5))}

    def delete_search_data(self, results):
        """
        After showing the results in the dashboards, removing .csv files from the temporary folder path.
        """
        if results['has_results']:
            for i in range(1, 5):
                _file = join(self.temporary_path, "build_in_reports", "main", 'chart_{}_search.csv'.format(str(i)))
                try:
                    os.unlink(_file)
                except Exception as e:
                    print("no file is observed!!!")








