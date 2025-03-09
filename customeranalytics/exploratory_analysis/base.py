import pandas as pd
import numpy as np

from customeranalytics.data_storage_configurations.connection import Connection


class BaseEDA(Connection):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_sets: dict[str, pd.DataFrame] = {}
        self.dimension_kpis = pd.DataFrame()
        self.daily_dimension_values = pd.DataFrame()
        self.orders_field_data = []
        self.download_field_data = []
        self.configs = kwargs

    def get_time_period(self, transactions, date_column):
        """
        converting date column of  values into the time_periods (hourly weekly, monthly,..)
        :param transactions: total data (orders/downloads data with actions)
        :return: data set with time periods
        """
        for p in list(zip(
                self.time_periods,
                [
                    self.convert_str_to_hour,
                    self.convert_dt_to_day_str,
                    self.find_week_of_monday,
                    self.convert_dt_to_month_str
                ])):
            transactions[p[0]] = transactions[date_column].apply(lambda x: p[1](x))
        return transactions

    def dimensional_query(self, boolean_query=None):
        if self.dimension_decision(self.order_index):
            if boolean_query is None:
                boolean_query = [{"term": {"dimension": self.order_index}}]
            else:
                boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def report_file_name(self, report_name, eda_type, start_date=None):
        report_date = (
            str(self.current_date_to_day().isoformat() if start_date is None else start_date)
            [:10]
        )
        file_name = f"{report_name}_{eda_type}_{report_date}"
        return file_name

    def get_data(self, start_date=None):
        """
        query orders index to collect the data with columns that are
        "id", "session_start_date", "client", "payment_amount", "discount_amount", "actions.purchased".
        :param start_date: starting date of query
        :return: data-frame individual order transactions.
        """
        start_date = self.default_query_date if start_date is None else start_date
        for data_type in self.configs.get('data_sets'):
            data = self.get_data_from_folder(data_type)

            if data_type == 'orders':
                data['date'] = data['session_start_date'].apply(self.convert_to_date)
                data = data.query(f"date >= {start_date}")
            if data_type == 'downloads':
                data['download_date'] = data['download_date'].apply(self.convert_to_date)
            self.data_sets[data_type] = data

    def create_report_data(
            self,
            report_name,
            eda,
            start_date,
            eda_type,
    ):
        """
        each report can be inserted into the folder which is assigned from Data Storage Configuration <data-conf.htm>
        :param report_name: churn funnel, stats
        :param eda: pandas data frame
        :param start_date: datetime
        :param eda_type: {"type": "overall" or "weekly_orders" or "daily_orders" or "monthly_orders"}
        """
        self.insert_data(
            eda,
            self.report_file_name(report_name, eda_type, start_date)
        )

    def fetch(
            self,
            report_name,
            eda_type,
            start_date=None,
            end_date=None,
            time_period=None,
            _from=None,
            _to=None,

    ):
        """
        query format;
            queries = {"churn_type": "overall"}
            queries = {"churn_type": "weekly"}
            queries = {"churn_type": "monthly"}
            	weekly	            churn
            0	2020-12-07T00:00:00	0.2
            1	2020-12-14T00:00:00	0.4
            2	2020-12-21T00:00:00	0.3
        :param eda_type:  overall, weekly, monthly
        :param start_date:
        :return: data-frame
        """
        file_name = self.report_file_name(report_name, eda_type, start_date)
        return self.get_data_from_folder(file_name)

        eda_type = {"report_types": {"type": eda_type}}
        if time_period is not None:
            eda_type['report_types']['time_period'] = time_period
        if _from is not None:
            eda_type['report_types']['_from'] = _from
        if _to is not None:
            eda_type['report_types']['to'] = _to
        boolean_queries = [{"term": {"report_name": report_name}},
                           {"term": eda_type},
                           {"term": {"index": self.get_index_group(self.order_index)}}]

        date_queries = []

        self.folder_file_list


        if start_date is not None:
            date_queries = [{"range": {"report_date": {"gte": self.convert_to_iso_format(start_date)}}}]
        self.query_es.query_builder(fields=None, _source=True,
                                    boolean_queries=boolean_queries,
                                    date_queries=date_queries)
        _res = self.query_es.get_data_from_es(index="reports")
        _data = pd.DataFrame()
        if len(_res) != 0:
            _data = pd.DataFrame(_res[0]['_source']['data'])
        return _data

