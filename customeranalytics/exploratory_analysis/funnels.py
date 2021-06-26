import numpy as np
import pandas as pd
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_es_port, default_es_host, elasticsearch_settings, time_periods, default_query_date
from customeranalytics.utils import *
from customeranalytics.data_storage_configurations.query_es import QueryES


class Funnels:
    """
        It is useful for creating an exploratory analysis of data frames for charts and tables.
            -   Purchase Actions Funnel:
                This funnel shows the values from session to purchase.
                ElasticSearch index must contain 'has_sessions' and 'purchased' actions.
                Additional actions must be assigned as arguments with 'purchase_actions'.
                There are other possible actions for purchase which are;
                    - has_basket
                    - has_sessions.
                These are available with sample (sample order index).

            -   Download Actions Funnel:
                This funnel shows the values from download to session.
                !!! It is not a funnel client who has download in the same time period (daily, weekly, ..) ordered.

        Each funnel is created per time period (daily, weekly, hourly, monthly)
        There are 3*2 and an overall funnel (total numbers) created.
        These funnels are stored in the elasticsearch reports index.
        It is flexible to collect funnel by using fetch.
    """
    def __init__(self,
                 actions,
                 purchase_actions=None,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        !!!!
        ******* ******** *****
        Dimensional Funnel:
        Funnels must be created individually for dimensions. For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimension can be assigned in order to create a funnel

        download_index; downloads_location1 this will be the location dimension of
                        parameters in order to query downloads indexes; 'location1'.
        download_index; orders_location1 this will be the location dimension of
                        parameters in order to query orders indexes; 'location1'.
        ******* ******** *****
        !!!

        :param actions: download additional actions
        :param purchase_actions: purchase additional actions
        :param host: elasticsearch host
        :param port: elasticsearch port
        :param download_index: elasticsearch port
        :param order_index: elasticsearch port
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.download_index = download_index
        self.order_index = order_index
        self.query_es = QueryES(port=self.port, host=self.port)
        self.actions = actions
        self.purchase_actions = purchase_actions
        self.purchase_action_funnel_data = {
                                   'purchased': {t: None for t in time_periods},
                                   'has_sessions': {t: None for t in time_periods}
                      }
        self.action_funnel_data = {
                                   'purchased': {t: None for t in time_periods},
                                   'has_sessions': {t: None for t in time_periods}
                                   }

        self.purchase_funnels = {t: None for t in time_periods}
        self.download_funnels = {t: None for t in time_periods}
        self.overall_funnels = {}
        self.time_periods = time_periods
        self.purchase_action_funnel_fields = ["id", "session_start_date"]
        self.action_funnel_fields = ["id", "session_start_date"]
        self.query_size = elasticsearch_settings['settings']['index.query.default_field']

    def get_time_period(self, transactions, date_column):
        """
        converting date column of  values into the time_periods (hourly weekly, monthly,..)
        :param transactions: total data (orders/downloads data with actions)
        :return: data set with time periods
        """
        for p in list(zip(self.time_periods,
                     [convert_str_to_hour, convert_dt_to_day_str, find_week_of_monday, convert_dt_to_month_str])):
            transactions[p[0]] = transactions[date_column].apply(lambda x: p[1](x))
        return transactions

    def dimensional_query(self, boolean_query=None):
        if dimension_decision(self.order_index):
            if boolean_query is None:
                boolean_query = [{"term": {"dimension": self.order_index}}]
            else:
                boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def get_purchase_action_funnel_data(self, action, date_column):
        """
        Each action has been calculated related to time periods.
        This process is an aggregation of related order id or client according to time periods.
        There are 4 time periods;
            - monthly
            - weekly
            - daily
            - hourly
        Aggregated value is assigned as a column related to the 'action' argument.
        date_column shows which date column must be used.
        There are 2 options for query the data by using query_es.py.
        Orders or Downloads indexes can be queried related to action.

        if it is calculating for dimensional data. ElasticSerachquery must have additional filter as below;
            {"term": {"dimension": self.order_index}}

        :param action: action from orders / downloads
        :param date_column: action_date column. If it is for orders, one option; session_start_date
        :param index: action from orders / downloads
        """
        transactions = pd.DataFrame(self.query_es.get_data_from_es())
        transactions = self.get_time_period(transactions=transactions, date_column=date_column)
        # hourly orders
        funnels = {'hourly': transactions.groupby(["daily", "hourly"]).agg({"id": "count"}).reset_index()}
        funnels['hourly'] = funnels['hourly'].groupby("hourly").agg({"id": "mean"}).reset_index().rename(
            columns={"id": "hourly_" + action})
        # weekly - monthly - daily orders
        for p in ['weekly', 'monthly', 'daily']:
            funnels[p] = transactions.groupby(p).agg(
                {"id": "count"}).reset_index().rename(columns={"id": "_".join([p, action])})
        return funnels

    def merge_actions(self, data):
        """
        After each action of calculation has been done related to the time period,
        The need to be merge related to time periods.
        After merging it is clearly seen the conversion from one action to another
        :param data: transaction data
        :return: funnels  - {'has_basket': {'hourly': dataframe, 'daily': dataframe, .... }, 'has_Sessions': {...}}
        """
        funnels = {}
        for p in self.time_periods:  # collects funnels
            _funnel = pd.DataFrame()
            for a in data:  # merge actions of dataframes
                if len(_funnel) != 0:
                    _funnel = pd.merge(_funnel, data[a][p], on=p, how='left')
                else: _funnel = data[a][p]
            funnels[p] = _funnel
        return funnels

    def purchase_action_funnel(self, start_date=None):
        """
        This is a funnel related to actions during a session. It starts from the session end with the purchase.
        -   All date calculations are applied via session_start_date.
        -   In order to calculate actions, Each action must be assigned to the orders index
            as a Boolean format in the 'actions' dictionary. e.g; actions.has_basket: True
        -   Each action must be queried individually via 'session_start_date'.
            In the query, only the boolean action will change.
        -   Example of purchased action query;

                    {'size': 10000000,
                     'from': 0,
                     '_source': False,
                     'fields': ['id', 'session_start_date'],
                     'query': {'bool': {'must': [{'term': {'actions.purchased': True}}]}}}

            Size is copied from elasticsearch orders index settings.
             No need for 'session_start_date' if it is calculating of whole orders index population.
             After funnels are ready, they are inserting into the reports index individually.
             Each time period of the funnel is inserted as an object into the reports index.
             - Example of inserting funnel;

                 {"id": 1232421424,
                  "report_date": '2021-01-01T00:00:00',
                  "report_name": "funnel",
                  "report_types": {"time_period": "daily, "type": 'orders'},
                  "data": [{'daily': Timestamp('2020-12-13 00:00:00'),
                            'daily_purchased': 3,
                            'daily_has_sessions': 1233,
                            'daily_has_basket': 1080,
                            'daily_order_screen': 269}]}

            - purchase funnel process:
                1. check if there are additional actions
                2. order session and other actions of count per week, day, hour (query_es.py - QueryBuilder)
                3. merge actions related to time periods
                4. insert into the reports index.

            - Required actions: session_start_date; datetime, has_session; boolean, has_purchase; boolean
        """

        if self.purchase_actions is not None:  # check for additional actions
            for a in self.purchase_actions:
                self.purchase_action_funnel_data[a] = {p: pd.DataFrame() for p in self.time_periods}

        for a in self.purchase_action_funnel_data:  # order session and other actions of count per week, day, hour
            self.query_es = QueryES(port=self.port, host=self.host)
            if start_date is not None:
                self.query_es.date_queries_builder({"session_start_date": {"gte": start_date}})
            self.query_es.query_builder(fields=self.purchase_action_funnel_fields,
                                        boolean_queries=self.dimensional_query([{"term": {"actions." + a: True}}]))
            self.purchase_action_funnel_data[a] = self.get_purchase_action_funnel_data(action=a,
                                                                                       date_column='session_start_date')
        # merge actions related to time periods
        self.purchase_funnels = self.merge_actions(self.purchase_action_funnel_data)
        # insert into the reports index
        self.insert_into_reports_index(self.purchase_funnels, start_date, index=self.order_index)

    def download_signup_session_order_funnel(self, start_date=None):
        """
        This is a funnel related to actions from download to purchase includes sessions.
        -   All date calculations are applied via session_start_date, download date, signup date or
            action + '_date'.
        -   In order to calculate actions, Each action must be assigned to downloads and orders index.
            Actions that are related to purchase, must be assigned as Boolean format in the 'actions' dictionary.
            e.g; actions.has_basket: True
            Actions that are related to downloads must be assigned to the downloads index as a date.
            If there is no action event that happens it must 'None' date value.
        -   Orders action 'sessions' and 'purchase' are pulled from purchase_funnel_data.
            Other actions must be queried individually from the downloads index.
        -   Example of Download action query;

                    {'size': 10000000,
                     'from': 0,
                     '_source': False,
                     'fields': ['id',  action +'_date'],
                     'query': {"bool": {"must": [{'range': {action +'_date': {'gte': start_date}}}]}}
                    }

             Size is copied from elasticsearch orders index settings.
             No need for 'session_start_date' if it is calculating of whole orders index population.
             After funnels are ready, they are inserting into the reports index individually.
             Each time period of the funnel is inserted as an object into the reports index.
             - Example of inserting funnel;

                 {"id": 1232421424,
                  "report_date": '2021-01-01T00:00:00',
                  "report_name": "funnel",
                  "report_types": {"time_period": "daily, "type": 'downloads'},
                  "data": [{'daily': Timestamp('2020-12-13 00:00:00'),
                            'daily_purchased': 3,
                            'daily_has_sessions': 1233,
                            'daily_download': 15824,
                            'daily_signup': 610}]}

            - download funnel process:
                1. check if there are additional actions
                2. download and other actions of count per week, day, hour (query_es.py - QueryBuilder)
                3. merge actions related to time periods
                4. insert into the reports index.

            - Required actions: download date; datetime, has_session; boolean, has_purchase; boolean
        """
        start_date = default_query_date if start_date is None else start_date
        if self.actions is not None:  # check for additional actions
            for a in self.actions:
                self.action_funnel_data[a] = {p: pd.DataFrame() for p in self.time_periods}

        for a in self.action_funnel_data:  # download and other actions of count per week, day, hour
            if a in ['purchased', 'has_sessions']:
                self.action_funnel_data[a] = self.purchase_action_funnel_data[a]
            else:
                _date_column = a + "_date"
                self.action_funnel_fields = ["id", _date_column]
                self.query_es = QueryES(port=self.port, host=self.host)
                self.query_es.date_queries_builder({_date_column: {"gte": start_date}})
                self.query_es.query_builder(fields=self.action_funnel_fields,
                                            boolean_queries=self.dimensional_query())
                self.action_funnel_data[a] = self.get_purchase_action_funnel_data(action=a,
                                                                                  date_column=_date_column)
        # merge actions related to time periods
        self.download_funnels = self.merge_actions(self.action_funnel_data)
        # insert into the reports index
        self.insert_into_reports_index(self.download_funnels, current_date_to_day().isoformat(),
                                       funnel_type='downloads', index=self.order_index)

    def overall_funnel(self, start_date=None):
        """
        All actions are combined into a dictionary. Total values are calculated via monthly funnels.
        1 row of the data frame is created.
        yearly arguments assigned for each action column. e.g; downloads = yearly_downloads
        overall_funnel is inserted into the reports index.
        :param start_date:
        :param index: refers the dimensionality of the whole data.
        """
        start_date = default_query_date if start_date is None else start_date
        if start_date is not None:  # filter monthly column for each actions
            dfs = []
            for df in [self.purchase_funnels['monthly'], self.download_funnels['monthly']]:
                dfs.append(df.query("monthly >= @start_date"))
            self.purchase_funnels['monthly'], self.download_funnels['monthly'] = dfs

        _action_columns = set( # collect all unique actions via orders and downloads actions
            list(self.purchase_funnels['monthly'].columns) +
            list(self.download_funnels['monthly'].columns)) - set(['monthly'])
        for a in _action_columns: # calculate each action of total
            _column = "_".join(['yearly', a.split("monthly_")[-1]])
            if a in list(self.purchase_funnels['monthly'].columns):
                self.overall_funnels[_column] = sum(self.purchase_funnels['monthly'][a])
            else:
                self.overall_funnels[_column] = sum(self.download_funnels['monthly'][a])
        self.time_periods = ['yearly']
        # insert into the reports index
        self.insert_into_reports_index({"yearly": pd.DataFrame([self.overall_funnels])},
                                       current_date_to_day().isoformat(), funnel_type='overall',
                                       index=self.order_index)
        self.time_periods = time_periods

    def insert_into_reports_index(self, funnel, start_date, funnel_type='orders', index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": start_date or current date,
         "report_name": "funnel",
         "index": "main",
         "report_types": {"time_period": yearly (only for overall funnel), monthly, hourly, weekly, daily
                          "type": orders, downloads, overall
                          },
         "data": funnel[t].to_dict("results") -  dataframe to list of dictionary
         }
        :param funnel: data set, data frame
        :param start_date: data start date
        :param funnel_type: orders, downloads
        :param index: dimentionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = []
        for t in self.time_periods:
            insert_obj = {"id": np.random.randint(200000000),
                          "report_date": current_date_to_day().isoformat() if start_date is None else start_date,
                          "report_name": "funnel",
                          "index": get_index_group(index),
                          "report_types": {"time_period": t, "type": funnel_type},
                          "data": funnel[t].fillna(0).to_dict("results")}
            list_of_obj.append(insert_obj)
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def fetch(self, funnel_name, start_date=None, end_date=None):
        """
        This allows us to query the created funnels.
        funnel_name is crucial for us to collect the correct filters.
        Example of queries;
            -   funnel name: funnel_downloads_daily,
            -   start_date: 2021-01-01T00:00:00
            -   end_date: 2021-04-01T00:00:00

            {'size': 10000000,
            'from': 0,
            '_source': True,
            'query': {'bool': {'must': [
                                        {'term': {'report_name': 'funnel'}},
                                        {"term": {"index": "orders_location1"}}
                                        {'term': {'report_types.time_period': 'daily'}},
                                        {'term': {'report_types.type': 'downloads'}},
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













