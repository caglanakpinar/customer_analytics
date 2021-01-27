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

from configs import default_es_port, default_es_host, elasticsearch_settings
from utils import convert_to_date, find_week_of_monday, convert_str_to_hour, convert_dt_to_day_str, calculate_time_diff
from data_storage_configurations.query_es import QueryES


class Cohorts:
    """

    """
    def __init__(self, has_download=True, host=None, port=None):
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.query_es = QueryES(port=port, host=host)
        self.has_download = has_download
        self.download_field_data = ["id", "download_date", "client"]
        self.session_orders_field_data = ["id", "session_start_date", "client"]
        self.downloads = pd.DataFrame()
        self.orders = pd.DataFrame()
        self.sessions = pd.DataFrame()
        self.time_periods = ['daily', 'weekly']
        self.cohorts = {'downloads_to_1st_order': {_t: None for _t in self.time_periods},
                        'orders_from_1_to_2': {_t: None for _t in self.time_periods},
                        'orders_from_2_to_3': {_t: None for _t in self.time_periods},
                        'orders_from_3_to_4': {_t: None for _t in self.time_periods}
                        }

        self.order_seq = [1, 2, 3]

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

    def get_data(self, start_date):
        """

        :param start_date:
        :return:
        """
        if len(self.orders) == 0:
            self.query_es = QueryES(port=self.port, host=self.port)
            self.query_es.query_builder(fields=self.session_orders_field_data,
                                        boolean_queries=[{"actions.purchased": True}],
                                        date_queries=[{"range": {"session_start_date": {"gte": start_date}}}])
            self.orders = self.query_es.get_data_from_es()
            self.orders = self.get_time_period(self.orders, 'session_start_date')
        if len(self.downloads) == 0:
            if self.has_download:
                self.query_es = QueryES(port=self.port, host=self.port)
                self.query_es.query_builder(fields=self.download_field_data)
                self.downloads = self.query_es.get_data_from_es(index='downloads')
                self.downloads = self.get_time_period(self.downloads, 'download_date')

    def convert_cohort_to_readable_form(self, cohort, time_period_back=None):
        """
        :param cohort:
        :return:
        """
        _time_periods = pd.DataFrame(list(cohort[cohort.columns[0]])).rename(columns={0: "days"})
        _cohort = pd.DataFrame(np.array(cohort.drop(cohort.columns[0], axis=1)))

        if time_period_back is not None:
            time_period_columns, time_period_row = time_period_back, time_period_back
            if max(list(_cohort.columns)) < time_period_back:
                time_period_columns = max(list(_cohort.columns))

            if len(_time_periods) < time_period_back:
                time_period_row = max(list(_cohort.columns))

            _cohort = _cohort[range(time_period_columns)]
            cohort = pd.concat([_time_periods, _cohort], axis=1).tail(time_period_row)
        else:
            cohort = pd.concat([_time_periods, _cohort], axis=1)
        return cohort

    def cohort_download_to_1st_order(self):
        """

        :return:
        """
        if self.has_download:
            self.download_to_first_order = pd.merge(self.orders.drop(['weeks', 'days', 'hours'], axis=1),
                                                    self.downloads,
                                                    on='client',
                                                    how='left')
            self.download_to_first_order['download_to_first_order_weekly'] = self.download_to_first_order.apply(
                lambda row: calculate_time_diff(row['download_date'], row['date'], period='week'), axis=1)
            self.download_to_first_order['download_to_first_order_daily'] = self.download_to_first_order.apply(
                lambda row: calculate_time_diff(row['download_date'], row['date'], period='day'), axis=1)

            for p in self.time_periods:
                self.cohorts['download_to_1st_order'][p] = self.download_to_first_order.sort_values(
                by=['download_to_first_order_' + p, p],
                ascending=True).pivot_table(index=p,
                                            columns='download_to_first_order_'+p,
                                            aggfunc={"client": lambda x: len(np.unique(x))}
                                            ).reset_index().rename(columns={"client": "client_count"})

                self.cohorts['download_to_1st_order'][p] = self.convert_cohort_to_readable_form(
                self.cohorts['download_to_1st_order'][p])

    def get_order_cohort(self, order_seq_num, time_period='daily'):
        index_column = time_period if time_period == 'daily' else 'weekly'
        column_pv = 'diff_days' if time_period == 'daily' else 'diff_weeks'
        orders_from_to = self.orders.query("next_order_date == next_order_date")
        orders_from_to = orders_from_to.query("order_seq_num in @order_seq_num")
        orders_from_to = orders_from_to.sort_values(by=[column_pv, index_column], ascending=True)
        orders_from_to = orders_from_to.pivot_table(index=index_column, columns=column_pv, aggfunc={
            "client": lambda x: len(np.unique(x))}).reset_index().rename(columns={"client": "client_count"})
        return orders_from_to

    def cohort_time_difference_and_order_sequence(self):
        if 'order_seq_num' not in self.orders.columns:
            self.orders['order_seq_num'] = \
            self.orders.sort_values(by=['client', 'date'], ascending=True).groupby(['client'])['client'].cumcount() + 1
        if 'next_order_date' not in self.orders.columns:
            self.orders['next_order_date'] = \
            self.orders.sort_values(by=['client', 'date'], ascending=True).groupby(['client'])['date'].shift(-1)
        if 'diff_days' not in self.orders.columns:
            self.orders['diff_days'] = self.orders.apply(
                lambda row: calculate_time_diff(row['date'], row['next_order_date'], 'day'), axis=1)
        if 'diff_weeks' not in self.orders.columns:
            self.orders['diff_weeks'] = self.orders.apply(
                lambda row: calculate_time_diff(row['date'], row['next_order_date'], 'week'), axis=1)

    def cohort_from_to_order(self):
        """

        :return:
        """
        for o in self.order_seq:
            for p in self.time_periods:
                self.cohort_time_difference_and_order_sequence()
                self.cohorts["order_from_" + str(o) + "_to_" + str(o+1)][p] = self.get_order_cohort(order_seq_num=[o],
                                                                                                    time_period=p)
                self.cohorts["order_from_" + str(o) + "_to_" + str(o+1)][p] = self.convert_cohort_to_readable_form(
                    self.cohorts["order_from_" + str(o) + "_to_" + str(o+1)][p])

    def customer_average_journey(self):
        """
        - Calculate Customers average total orders
        - Line chart from download to first order average hour difference
        - Iteratively add x axis the average time difference from 1 order to next one, till the average total order count.
        - Y axis will be average amount per order

        """
        self.cohorts['customer_average_journey'] = pd.DataFrame()

    def insert_into_reports_index(self, cohort, start_date, _from=0, _to=1, cohort_type='orders'):
        """
        :return:
        """
        list_of_obj = []
        for t in self.time_periods:
            insert_obj = {"id": np.random.randint(200000000),
                          "report_date": current_date_to_day().isoformat() if start_date is None else start_date,
                          "report_name": "cohort",
                          "report_types": {"time_period": t, "from": _from, "to": _to,  "type": cohort_type},
                          "data": cohort[t].to_dict("results")}
            list_of_obj.append(insert_obj)
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def get_cohort_name(self, cohort_name):
        _cohort_type = cohort_name.split("_")[0]
        if _cohort_type == 'order':
            _from, _to = cohort_name.split("_")[2], cohort_name.split("_")[3]
        else:
            _from, _to = 0, 1
        return _cohort_type, _from, _to

    def execute_cohort(self, start_date):
        """

        :return:
        """
        self.get_data(start_date)
        self.cohort_download_to_1st_order()
        self.cohort_from_to_order()
        self.customer_average_journey()

        for _c in self.cohorts:
            for p in self.time_periods:
                _cohort_type, _from, _to = self.get_cohort_name(_c)
                self.insert_into_reports_index(self.cohorts[_c],
                                               start_date,
                                               _from=_from,
                                               _to=_to,
                                               cohort_type=_cohort_type)

        self.time_periods = ['yearly']
        self.insert_into_reports_index(self.cohorts['customer_average_journey'],
                                       start_date,
                                       _from=0,
                                       _to=100,
                                       cohort_type='customer_journey')
        self.time_periods = ['yearly']

    def fetch(self, cohort_name, _from=None, _to=None, start_date=None, end_date=None):
        """

        :return: data frame
        """
        _cohort_type, _from, _to = self.get_cohort_name(cohort_name)
        _time_period = cohort_name.split("_")[-1]
        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_name": "cohort"}},
                           {"term": {"report_types.time_period": _time_period}},
                           {"term": {"report_types.type": _cohort_type}},
                           {"term": {"report_types._from": _from}},
                           {"term": {"report_types._to": _to}}]

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
                _data[_time_period] = _data[_time_period].apply(lambda x: convert_to_date(x))
                if _time_period not in ['yearly', 'hourly']:
                    start_date = convert_to_date(start_date)
                    _data = _data[_data[_time_period] >= start_date]
        return _data













