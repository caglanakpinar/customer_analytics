import numpy as np
import pandas as pd
import sys, os, inspect
import warnings
warnings.filterwarnings("ignore")

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_es_port, default_es_host, default_query_date, time_periods
from customeranalytics.utils import *
from customeranalytics.data_storage_configurations.query_es import QueryES


class Churn:
    """
    Churn is the crucial KPI for the businesses. While they are tracking the engaged customers,
    they also check the user who have been lost and never use the business anymore.

    Here are the basic aggregated values;
        -   Overall Churn rate
        -   Weekly Churn Rate
        -   Monthly Churn Rate

    !!!!
    ******* ******** *****
    Dimensional Churn:
    Churn Rate must be created individually for dimensions.
    For instance, the Data set contains locations dimension.
    In this case, each location of 'orders' and 'downloads' indexes must be created individually.
    by using 'download_index' and 'order_index' dimension can be assigned in order to create the Churn Rate.

    download_index; downloads_location1 this will be the location dimension of
                    parameters in order to query downloads indexes; 'location1'.
    download_index; orders_location1 this will be the location dimension of
                    parameters in order to query orders indexes; 'location1'.
    ******* ******** *****
        !!!
    """

    def __init__(self,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        ******* ******** *****
        Dimensional Stats:
        Descriptive Statistics must be created individually for dimensions.
        For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimension can be assigned in order to create the Churn Rate.

        download_index; downloads_location1 this will be the location dimension of
                        parameters in order to query downloads indexes; 'location1'.
        download_index; orders_location1 this will be the location dimension of
                        parameters in order to query orders indexes; 'location1'.
        ******* ******** *****
        !!!

        :param host: elasticsearch host
        :param port: elasticsearch port
        :param download_index: elasticsearch port
        :param order_index: elasticsearch port
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.download_index = download_index
        self.order_index = order_index
        self.query_es = QueryES(port=port, host=host)
        self.orders_field_data = ["id", "session_start_date", "client", "actions.purchased"]
        self.last_week = None
        self.time_periods = ["weekly", 'monthly']
        self.orders = pd.DataFrame()
        self.results = {}
        self.average_frequency_hr = 7 * 24

    def get_time_period(self, transactions, date_column):
        """
        converting date column of  values into the time_periods (hourly weekly, monthly,..)
        :param transactions: total data (orders/downloads data with actions)
        :return: data set with time periods
        """
        for p in list(zip(self.time_periods,
                     [find_week_of_monday, convert_dt_to_month_str])):
            transactions[p[0]] = transactions[date_column].apply(lambda x: p[1](x))
        return transactions

    def dimensional_query(self, boolean_query=None):
        if dimension_decision(self.order_index):
            if boolean_query is None:
                boolean_query = [{"term": {"dimension": self.order_index}}]
            else:
                boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def get_data(self, start_date=None):
        """
        query orders index to collect the data with columns that are
        "id", "session_start_date", "client", "payment_amount", "discount_amount", "actions.purchased".
        :param start_date: starting date of query
        :return: data-frame individual order transactions.
        """
        start_date = default_query_date if start_date is None else start_date
        if len(self.orders) == 0:
            self.query_es = QueryES(port=self.port, host=self.host)
            self.query_es.query_builder(fields=self.orders_field_data,
                                        date_queries=[{"range": {"session_start_date": {"gte": start_date}}}],
                                        boolean_queries=self.dimensional_query())
            self.orders = pd.DataFrame(self.query_es.get_data_from_es())
            self.orders['date'] = self.orders['session_start_date'].apply(lambda x: convert_to_date(x))

    def frequency(self):
        """
        Frequency of users;
            -   assign dates of next orders per user as a column.
                So, each row will have a current order date and the next order date per user.
            -   Calculate the hour difference from the current order date to the next order date.
            -   Calculate the average hourly difference per user.
        User has only 1 order will not be included in calculations.
        """
        self.orders['next_order_date'] = self.orders.sort_values(
            by=['client', 'date'], ascending=True).groupby(['client'])['date'].shift(-1)
        self.orders['diff_hours'] = self.orders.apply(
            lambda row: calculate_time_diff(row['date'], row['next_order_date'], 'hour'), axis=1)
        _fequency = self.orders.query("next_order_date == next_order_date").groupby("client").agg(
            {"diff_hours": "mean"}).reset_index().rename(columns={"diff_hours": "frequency"})
        self.average_frequency_hr = int(np.mean(_fequency['frequency']))

    def churn_rate(self):
        """
        (Unique user count who have at least 1 order -
         Unique Order who has order at least 1 order during days between last average_frequency_hr and current day)
        /
         (Count of  Unique Client who has at least 1 order)
        """
        _customer_loss_date = max(self.orders['date']) - datetime.timedelta(hours=self.average_frequency_hr)
        _purc_orders = self.orders[self.orders['actions.purchased'] == True]
        engaged_users = len(list(_purc_orders[_purc_orders['date'] >= _customer_loss_date]['client'].unique()))
        all_users = len(list(_purc_orders['client'].unique()))
        _churn = (all_users - engaged_users) / all_users
        return pd.DataFrame([{"churn": _churn if _churn > 0 else 0}])

    def churn_rate_per_time_period(self, time_period):
        """
        Churn rate per week and per month
        each week/month of unique ordered client counts are calculated and
        divided by cumulative sum of weekly/monthly unique client count.
        """

        _purc_clients = self.orders[self.orders['actions.purchased'] == True].groupby(time_period).agg(
            {"client": lambda x: len(np.unique(x))}).reset_index().rename(
            columns={"client": "client_count"}).sort_values(time_period, ascending=True)
        _purc_clients['weekly_whole_orders'] = list(_purc_clients['client_count'].cumsum())
        _purc_clients['churn'] = (_purc_clients['weekly_whole_orders'] - _purc_clients['client_count']) / _purc_clients[
            'weekly_whole_orders']
        print(_purc_clients)
        _purc_clients['churn'] = _purc_clients['churn'].apply(lambda x: x if x > 0 else 0)
        return _purc_clients[[time_period, 'churn']]

    def execute_churn(self, start_date):
        """
        1. Collect users with action.purchased column
        2. assign weekly/monthly column
        3. calculate overall churn rate
        4. calculate weekly/monthly churn rate
        """
        self.get_data()
        self.orders = self.get_time_period(self.orders, 'date')
        self.frequency()
        self.insert_into_reports_index(self.churn_rate(), start_date, 'overall')
        for tp in self.time_periods:
            self.insert_into_reports_index(self.churn_rate_per_time_period(tp), start_date, tp)

    def insert_into_reports_index(self, churn, start_date, churn_type, index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": start_date or current date,
         "report_name": "churn",
         "index": "main",
         "report_types": {
                          "type": "overall", "weekly", "monthly"
                          },
         "data": churn (list of dictionaries)
         }
        :param churn: overall, weekly, monthly
        :param start_date: datetime
        :param churn_type: {"type": "overall" or "weekly_orders" or "daily_orders" or "monthly_orders"}
        :param index: dimensionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = [{"id": np.random.randint(200000000),
                        "report_date": current_date_to_day().isoformat() if start_date is None else start_date,
                        "report_name": "churn",
                        "index": get_index_group(index),
                        "report_types": {"type": churn_type},
                        "data": churn.to_dict('results')}]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def fetch(self, churn_type, start_date=None):
        """
        query format;
            queries = {"churn_type": "overall"}
            queries = {"churn_type": "weekly"}
            queries = {"churn_type": "monthly"}
            	weekly	            churn
            0	2020-12-07T00:00:00	0.2
            1	2020-12-14T00:00:00	0.4
            2	2020-12-21T00:00:00	0.3
        :param churn_type:  overall, weekly, monthly
        :param start_date:
        :return: data-frame
        """

        boolean_queries = [{"term": {"report_name": "stats"}},
                           {"term": {"report_types.type": churn_type}},
                           {"term": {"index": get_index_group(self.order_index)}}]
        date_queries = []
        if start_date is not None:
            date_queries = [{"range": {"report_date": {"gte": convert_to_iso_format(start_date)}}}]

        self.query_es = QueryES(port=self.port,
                                host=self.host)
        self.query_es.query_builder(fields=None, _source=True,
                                    boolean_queries=boolean_queries,
                                    date_queries=date_queries)
        _res = self.query_es.get_data_from_es(index="reports")
        _data = pd.DataFrame()
        if len(_res) != 0:
            _data = pd.DataFrame(_res[0]['_source']['data'])
        return _data