import pandas as pd
import numpy as np

from customeranalytics.data_storage_configurations.connection import Connection


class BaseML(Connection):
    def __init__(self, host=None, port=None, download_index='downloads', order_index='orders'):
        super().__init__()
        self.download_index = download_index
        self.order_index = order_index
        self.port = self.default_es_port if port is None else port
        self.host = self.default_es_host if host is None else host
        self.orders = pd.DataFrame()
        self.downloads = pd.DataFrame()
        self.dimension_kpis = pd.DataFrame()
        self.daily_dimension_values = pd.DataFrame()
        self.orders_field_data = []
        self.download_field_data = []
        self.has_download = False

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

    def get_data(self, start_date=None):
        """
        query orders index to collect the data with columns that are
        "id", "session_start_date", "client", "payment_amount", "discount_amount", "actions.purchased".
        :param start_date: starting date of query
        :return: data-frame individual order transactions.
        """
        start_date = self.default_query_date if start_date is None else start_date

        for data_type in self.configs.get('data_sets'):
            self.data_sets[data_type] = self.get_data_from_folder(data_type)

            self.query_es.query_builder(fields=self.orders_field_data,
                                        date_queries=[{"range": {"session_start_date": {"gte": start_date}}}],
                                        boolean_queries=self.dimensional_query())
            self.orders = pd.DataFrame(self.query_es.get_data_from_es())
            self.orders['date'] = self.orders['session_start_date'].apply(self.convert_to_date)

        if len(self.downloads) == 0:
            if self.has_download:
                self.query_es.query_builder(fields=self.download_field_data)
                self.downloads = pd.DataFrame(self.query_es.get_data_from_es(index='downloads'))
                # for the dimensional it is only calculating for dimension of users.
                if self.dimension_decision(self.order_index):
                    self.downloads = self.downloads[self.downloads['client'].isin(list(self.orders['client'].unique()))]
                self.downloads = self.get_time_period(self.downloads, 'download_date')
                self.downloads['download_date'] = self.downloads['download_date'].apply(self.convert_to_date)

    def insert_into_reports_index(
            self,
            ml_name,
            ml,
            start_date,
            eda_type,
            end_data=None,
            _from=None,
            _to=None,
            time_period=None,
            index='orders'
    ):
        """
        via query.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": start_date or current date,
         "report_name": "churn",
         "index": "main",
         "report_types": {
                          "type": "overall", "weekly", "monthly"
                          },
         "data": churn (list of dictionaries)
         }
        :param eda: overall, weekly, monthly eda e.g. churn funnel, stats
        :param start_date: datetime
        :param eda_type: {"type": "overall" or "weekly_orders" or "daily_orders" or "monthly_orders"}
        :param index: dimensionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = [
            {"id": np.random.randint(200000000),
            "report_date": self.current_date_to_day().isoformat() if start_date is None else start_date,
            "report_name": ml_name,
            "index": self.get_index_group(index),
            "report_types": {"type": eda_type},
            "data": ml.to_dict('records')
             }
        ]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def fetch(
            self, ml_name, ml_type, start_date=None,  end_date=None,
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
        eda_type = {"report_types": {"type": ml_type}}
        if time_period is not None:
            eda_type['report_types']['time_period'] = time_period
        if _from is not None:
            eda_type['report_types']['_from'] = _from
        if _to is not None:
            eda_type['report_types']['to'] = _to
        boolean_queries = [{"term": {"report_name": ml_name}},
                           {"term": eda_type},
                           {"term": {"index": self.get_index_group(self.order_index)}}]
        date_queries = []
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

