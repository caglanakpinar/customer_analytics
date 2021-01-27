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

    def converting_days_weeks_hours(self, data, date_column):
        data['weekly'] = data[date_column].apply(lambda x: find_week_of_monday(x))
        data['hourly'] = data[date_column].apply(lambda x: convert_str_to_hour(x))
        data['daily'] = data[date_column].apply(lambda x: convert_dt_to_day_str(x))
        data[date_column] = data[date_column].apply(lambda x: convert_to_date(x))
        return data

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
            self.orders = self.converting_days_weeks_hours(self.orders, 'session_start_date')
        if len(self.downloads) == 0:
            if self.has_download:
                self.query_es = QueryES(port=self.port, host=self.port)
                self.query_es.query_builder(fields=self.download_field_data)
                self.downloads = self.query_es.get_data_from_es(index='downloads')
                self.downloads = self.converting_days_weeks_hours(self.downloads, 'download_date')

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
                self.cohorts['downloads_to_1st_order'][p] = self.download_to_first_order.sort_values(
                by=['download_to_first_order_weekly', p],
                ascending=True).pivot_table(index=p,
                                            columns='download_to_first_order_'+p,
                                            aggfunc={"client": lambda x: len(np.unique(x))}
                                            ).reset_index().rename(columns={"client": "client_count"})

                self.cohorts['downloads_to_1st_order'][p] = self.convert_cohort_to_readable_form(
                self.cohorts['downloads_to_1st_order'][p])

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

    def execute_cohort(self, start_date):
        """

        :return:
        """
        self.get_data(start_date)
        self.cohort_download_to_1st_order()
        self.cohort_from_to_order()

        for _c in self.cohorts:
            for p in self.time_periods:














