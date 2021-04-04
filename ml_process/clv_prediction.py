import sys, os, inspect
import pandas as pd
import numpy as np
import shutil
from clv.executor import CLV

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from configs import default_es_port, default_es_host, default_query_date
from utils import *
from data_storage_configurations.query_es import QueryES


class CLVPrediction:
    """
    Customer Lifetime Value Prediction;
    For more information pls check; https://github.com/caglanakpinar/clv_prediction


    """
    def __init__(self,
                 temporary_export_path,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        ******* ******** *****
        Dimensional CLV Prediction:
        Descriptive Statistics must be created individually for dimensions.
        For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimension can be assigned in order to create the CLV Prediction.

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
        self.clv = None
        self.query_es = QueryES(port=port, host=host)
        self.path = temporary_export_path
        self.temp_csv_file = join(temporary_export_path, "temp_data.csv")
        self.clv_fields = ["client", "session_start_date", "payment_amount"]
        self.clv_predictions = pd.DataFrame()

    def get_orders_data(self, end_date):
        """
        Purchased orders are collected from Orders Index.
        Session_start_date, client, payment_amount are needed for initializing CLv Prediciton.
        :param end_date: final date to start clv prediction
        :return:
        """
        self.query_es = QueryES(port=self.port, host=self.host)
        self.query_es.date_queries_builder({"session_start_date": {"lt": end_date}})
        self.query_es.boolean_queries_buildier({"actions.purchased": True})
        self.query_es.query_builder(fields=self.clv_fields)
        _data = pd.DataFrame(self.query_es.get_data_from_es())
        _data['session_start_date'] = _data['session_start_date'].apply(lambda x: str(convert_to_date(x)))
        _data.query("session_start_date == session_start_date").to_csv(self.temp_csv_file, index=False)

    def insert_into_reports_index(self, clv_predictions, time_period, date=None, index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": date or current date,
         "report_name": "clv_prediction",
         "index": "main",
         "report_types": {"time_period": weekly, monthly, daily },
         "data": segments.fillna(0.0).to_dict("results") -  dataframe to list of dictionary
         }
         !!! null values are assigned to 0.

        :param segments: data set, data frame
        :param index: dimensionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = [{"id": np.random.randint(200000000),
                        "report_date": current_date_to_day().isoformat() if date is None else date.isoformat(),
                        "report_name": "clv_prediction",
                        "index": get_index_group(index),
                        "report_types": {"time_period": time_period},
                        "data": clv_predictions.fillna(0.0).to_dict("results")}]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def convert_time_period(self, period):
        """
        clv_prediction library takes time-period as week, day, month.
        This converts the general time-period into the clv_prediction form.
        :param period: weekly or daily or month.
        :return: week or day or month
        """
        if period == 'weekly':
            return 'week'
        if period == 'daily':
            return 'day'
        if period == 'monthly':
            return 'month'

    def execute_clv(self, start_date, job='train', time_period='weekly'):
        """
        1.  train clv prediction models
            For more details about models pls check; https://github.com/caglanakpinar/clv_prediction
        2. predict users of feature payment amount via using built models.
        3. Each model and stored in the directory at elasticsearch folder. After it is used, it is removed.

        :param start_date: date of start for clv prediction
        :param job: train or prediction
        :param time_period: weekly, daily, monthly
        """
        start_date = str(current_date_to_day())[0:10] if start_date is None else start_date
        self.get_orders_data(end_date=start_date)
        self.clv = CLV(customer_indicator="client",
                       amount_indicator="payment_amount",
                       job=job,
                       date=start_date,
                       data_source='csv',
                       data_query_path=self.temp_csv_file,
                       time_period=self.convert_time_period(time_period),
                       time_indicator="session_start_date",
                       export_path=self.path)
        self.clv.clv_prediction()
        if job == 'train':
            self.clv = CLV(customer_indicator="client",
                           amount_indicator="payment_amount",
                           job='prediction',
                           date=start_date,
                           data_source='csv',
                           data_query_path=self.temp_csv_file,
                           time_period=self.convert_time_period(time_period),
                           time_indicator="session_start_date",
                           export_path=self.path)
            self.clv.clv_prediction()
        self.clv_predictions = \
        self.clv.get_result_data().query("data_type == 'prediction' and session_start_date == session_start_date")[
            ['session_start_date', 'client', 'payment_amount']]
        self.insert_into_reports_index(self.clv_predictions,
                                       time_period=time_period,
                                       date=start_date,
                                       index=self.order_index)
        try:
            os.unlink(self.temp_csv_file)
        except Exception as e:
            print("no file is observed!!!")

    def fetch(self, end_date=None, time_period='weekly'):
        """
        Collect CLV Prediction results with the given date.
        :param end_date: final date to start clv prediction
        :param time_period: weekly or daily or month.
        :return:
        """

        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_name": "clv_prediction"}},
                           {"term": {"report_types.time_period": time_period}},
                           {"term": {"index": get_index_group(self.order_index)}}]

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
            if end_date is not None:
                _data[time_period] = _data["session_start_date"].apply(lambda x: convert_to_date(x))
        return _data