import numpy as np
import pandas as pd
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from configs import default_es_port, default_es_host, elasticsearch_settings, time_periods, default_query_date
from utils import *
from data_storage_configurations.query_es import QueryES


class Funnels:
    def __init__(self, actions, purchase_actions=None, host=None, port=None):
        """
        It is useful for creating exploratory analysis of data frames for charts and tables.
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
                !!! It is not a funnel client who has download in same time period (daily, weekly, ..) ordered.

        Each funnel is craeted per time period (daily, weekly, hourly, monthly)
        There are 3*2 and overall funnel (total numbers) created.
        These funnels are stored in the elasticsearch reports index.
        It is flexiable to collect funnel by using fetch.

        :param actions: download additional actions
        :param purchase_actions: purchase additional actions
        :param host: elasticsearch host
        :param port: elasticsearch port
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
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
        self.downloads_fields = []
        self.match = {}
        self.query_size = elasticsearch_settings['settings']['index.query.default_field']

    def get_time_period(self, transactions, date_column):
        """
        converting date values into the time_periods (hourly weekly, monthly,..)
        :param transactions: total data (orders/downloads data with actions)
        :return: data set with time periods
        """
        transactions['hourly'] = transactions[date_column].apply(lambda x: convert_str_to_hour(x))
        transactions['daily'] = transactions[date_column].apply(lambda x: convert_dt_to_day_str(x))
        transactions['weekly'] = transactions[date_column].apply(lambda x: find_week_of_monday(x))
        transactions['monthly'] = transactions[date_column].apply(lambda x: convert_dt_to_month_str(x))
        return transactions

    def get_purchase_action_funnel_data(self, action, date_column, index='orders'):
        """
        Each action has been calculated related to time periods.
        This process is aggregation of related  order id or client according to time periods.
        There are 4 time periods;
            - monthly
            - weekly
            - daily
            - hourly
        Aggregated value is assigned as a column related to 'action' argument.
        date_column shows which date column must be used.
        There are 2 options for query the data by using query_es.py.
        Orders or Downloads indexes can be queried related to action.
        :param action: action from orders / downloads
        :return:
        """

        transactions = pd.DataFrame(self.query_es.get_data_from_es(index=index))
        transactions = self.get_time_period(transactions=transactions, date_column=date_column)

        # daily orders
        daily_t = transactions.groupby("daily").agg({"id": "count"}).reset_index().rename(
            columns={"id": "daily_" + action})
        # hourly orders
        hourly_t = transactions.groupby(["daily", "hourly"]).agg({"id": "count"}).reset_index()
        hourly_t = hourly_t.groupby("hourly").agg({"id": "mean"}).reset_index().rename(
            columns={"id": "hourly_" + action})
        # weekly orders
        weekly_t = transactions.groupby("weekly").agg({"id": "count"}).reset_index().rename(
            columns={"id": "weekly_" + action})
        # monthly orders
        monthly_t = transactions.groupby("monthly").agg({"id": "count"}).reset_index().rename(
            columns={"id": "monthly_" + action})
        return {"daily": daily_t, "hourly": hourly_t, "weekly": weekly_t, "monthly": monthly_t}

    def merge_actions(self, data):
        """
        After each action of calculation has been done related to time period,
        The need to be merge related to time periods.
        After merging it is clearly seen the conversion from one action to another
        :param data: transaction data
        :return:
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

        :return:
        """

        if self.purchase_actions is not None:  # check for additional actions
            for a in self.purchase_actions:
                self.purchase_action_funnel_data[a] = {p: pd.DataFrame() for p in self.time_periods}

        for a in self.purchase_action_funnel_data:  # order session and other actions of count per week, day, hour
            self.query_es = QueryES(port=self.port, host=self.host)
            if start_date is not None:
                self.query_es.date_queries_builder({"session_start_date": {"gte": start_date}})
            self.query_es.boolean_queries_buildier({"actions." + a: True})
            self.query_es.query_builder(fields=self.purchase_action_funnel_fields)
            print(self.query_es.match)
            self.purchase_action_funnel_data[a] = self.get_purchase_action_funnel_data(action=a,
                                                                                       date_column='session_start_date')
        self.purchase_funnels = self.merge_actions(self.purchase_action_funnel_data)
        self.insert_into_reports_index(self.purchase_funnels, start_date)

    def download_signup_session_order_funnel(self, start_date=None):
        """

        :return:
        """
        start_date = default_query_date if start_date is None else start_date
        if self.actions is not None:  # check for additional actions
            for a in self.actions:
                self.action_funnel_data[a] = {p: pd.DataFrame() for p in self.time_periods}
        for a in self.action_funnel_data:  # order session and other actions of count per week, day, hour
            if a in ['purchased', 'has_sessions']:
                self.action_funnel_data[a] = self.purchase_action_funnel_data[a]
            else:
                _date_column = a + "_date"
                self.action_funnel_fields = ["id", _date_column]
                self.query_es = QueryES(port=self.port, host=self.host)
                self.query_es.date_queries_builder({_date_column: {"gte": start_date}})
                self.query_es.query_builder(fields=self.action_funnel_fields)
                print(self.query_es.match)
                self.action_funnel_data[a] = self.get_purchase_action_funnel_data(action=a,
                                                                                  date_column=_date_column,
                                                                                  index='downloads')
        self.download_funnels = self.merge_actions(self.action_funnel_data)
        self.insert_into_reports_index(self.download_funnels, start_date, funnel_type='downloads')

    def overall_funnel(self, start_date=None):


        print(self.purchase_funnels['monthly'].head())
        print(self.download_funnels['monthly'].head())

        if start_date is not None:
            dfs = []
            for df in [self.purchase_funnels['monthly'], self.download_funnels['monthly']]:
                dfs.append(df.query("monthly >= @start_date"))
            self.purchase_funnels['monthly'], self.download_funnels['monthly'] = dfs

        _action_columns = set(
            list(self.purchase_funnels['monthly'].columns) +
            list(self.download_funnels['monthly'].columns)) - set(['monthly'])

        print(_action_columns)

        for a in _action_columns:
            _column = "_".join(['yearly', a.split("monthly_")[-1]])
            if a in self.purchase_funnels['monthly'].columns:
                self.overall_funnels[_column] = sum(self.purchase_funnels['monthly'][a])
            else:
                self.overall_funnels[_column] = sum(self.download_funnels['monthly'][a])
        print(pd.DataFrame([self.overall_funnels]))
        self.time_periods = ['yearly']
        self.insert_into_reports_index({"yearly": pd.DataFrame([self.overall_funnels])},
                                       start_date, funnel_type='overall')
        self.time_periods = time_periods

    def insert_into_reports_index(self, funnel, start_date, funnel_type='orders'):
        """
        :return:
        """
        list_of_obj = []
        for t in self.time_periods:
            insert_obj = {"id": np.random.randint(200000000),
                          "report_date": current_date_to_day().isoformat() if start_date is None else start_date,
                          "report_name": "funnel",
                          "report_types": {"time_period": t, "type": funnel_type},
                          "data": funnel[t].to_dict("results")}
            list_of_obj.append(insert_obj)
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def fetch(self, funnel_name, start_date=None, end_date=None):
        """

        :param funnel_name:
        :param start_date:
        :param end_data:
        :param orders:
        :return:
        """
        # funnel_name funnel_download_weekly
        report_name, funnel_type, time_period = funnel_name.split("_")
        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_name": report_name}},
                           {"term": {"report_types.time_period": time_period}},
                           {"term": {"report_types.type": funnel_type}}]

        if end_date is not None:
            date_queries = [{"range": {"report_date": {"lt": convert_to_iso_format(end_date)}}}]

        self.query_es = QueryES(port=self.port,
                                host=self.host)
        self.query_es.query_builder(fields=None, _source=True,
                                    date_queries=date_queries,
                                    boolean_queries=boolean_queries)
        _data = pd.DataFrame(self.query_es.get_data_from_es(index="reports")[0]['_source']['data'])

        if time_period not in ['yearly', 'hourly']:
            _data[time_period] = _data[time_period].apply(lambda x: convert_to_date(x))
            if start_date is not None:
                start_date = convert_to_date(start_date)
                _data = _data[_data[time_period] >= start_date]

        return _data













