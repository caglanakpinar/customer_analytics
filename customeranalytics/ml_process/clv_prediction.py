import pandas as pd
import numpy as np
import glob
import os
from clv.executor import CLV
from tomlkit import table

from customeranalytics.ml_process.base import BaseML
from customeranalytics.data_storage_configurations.reports import Reports


class CLVPrediction(BaseML):
    """
    Customer Lifetime Value Prediction;
    For more information pls check; https://github.com/caglanakpinar/clv_prediction


    """
    def __init__(self, temporary_export_path, host=None, port=None, time_period='weekly', download_index='downloads',
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
        super().__init__(host, port, download_index, order_index)
        self.download_index = download_index
        self.order_index = order_index
        self.time_period = time_period
        self.clv = None
        self.path = temporary_export_path
        self.temp_csv_file = os.path.join(temporary_export_path, "temp_data.csv")
        self.temp_folder = os.path.join(temporary_export_path, "temp_purchase_amount_results", "*.csv")
        self.results = os.path.join(temporary_export_path, "results_*.csv")
        self.clv_fields = ["client", "session_start_date", "payment_amount", "dimension"]
        self.result_query = "session_start_date == session_start_date and data_type != 'actual'"
        self.result_columns = ['session_start_date', 'client', 'payment_amount']
        self.clv_predictions = pd.DataFrame()
        self.client_dimensions = pd.DataFrame()
        self.collect_reports = Reports()
        self.has_prev_clv = False

    def get_customers_dimensions(self, _data):
        _data["dimension"] = _data["dimension"].fillna("main")
        if list(_data['dimension'].unique()) != 1:
            _data_pv = _data.groupby(["client", "dimension"]).agg({"session_start_date": "count"}).reset_index().rename(
                columns={"session_start_date": "order_count"})
            _data_pv['order_seq_num'] = _data_pv.sort_values(
                by=["client", "dimension", "order_count"], ascending=False).groupby(["client"]).cumcount() + 1
            _data_pv = _data_pv.query("order_seq_num == 1")
            _data_pv = _data_pv[["client", "dimension"]].groupby("client").agg({"dimension": "first"}).reset_index()
            self.client_dimensions = pd.merge(_data.drop('dimension', axis=1), _data_pv, on='client', how='left')

    def get_orders_data(self, end_date):
        """
        Purchased orders are collected from Orders Index.
        Session_start_date, client, payment_amount are needed for initializing CLv Prediciton.
        :param end_date: final date to start clv prediction
        :return:
        """
        self.query_es.date_queries_builder({"session_start_date": {"lt": end_date}})
        self.query_es.query_builder(fields=self.clv_fields,
                                    boolean_queries=[{"term": {"actions.purchased": True}}])
        _data = pd.DataFrame(self.query_es.get_data_from_es())
        _data['session_start_date'] = _data['session_start_date'].apply(lambda x: str(self.convert_to_date(x)))
        self.get_customers_dimensions(_data)
        _data.query("session_start_date == session_start_date").to_csv(self.temp_csv_file, index=False)

    def check_for_previous_clv_prediction(self):
        """
        check if the last clv prediction is applied in 7 days.
        """
        try:

            _reports = self.collect_reports.collect_reports(self.port, self.host, 'main',
                                                            query={"report_name": "clv_prediction"})
            if len(_reports) != 0:
                try:
                    _directory = self.collect_data_from_table(table="es_connection")[-1]['directory']
                    print(_directory)
                    clv_report = pd.read_csv(os.path.join(_directory, "build_in_reports", "main", "daily_clv.csv"))
                    if len(clv_report) != 0:
                        clv_report = clv_report.query("data_type == 'prediction'")
                        clv_report['date'] = clv_report['date'].apply(self.convert_to_day)
                        last_transaction_date = min(list(clv_report['date']))
                        print(last_transaction_date)
                except Exception as e:
                    print(e)
                    _reports['report_date'] = _reports['report_date'].apply(self.convert_to_date)
                    last_transaction_date = max(list(_reports['report_date']))

                if abs(last_transaction_date - self.current_date_to_day()).total_seconds() / 60 / 60 / 24 < 7:
                    print("clv predictions will be updated!!!!!")
                    self.has_prev_clv = True
        except: pass

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
        if period == '6 months':
            return '6*month'

    def remove_temp_files(self):
        """
        Removing temp_data.csv, .csv files in temp_purchase_amount_results and results*.csv files
        """
        # remove temp_data.csv
        try:
            os.unlink(self.temp_csv_file)
        except Exception as e:
            print("no file is observed!!!")

        # remove results*.csv files
        for f in glob.glob(self.results):
            print(f)
            try:
                os.unlink(f)
            except Exception as e:
                print(e)
                print("no file is observed!!!")

        # remove .csv files in temp_purchase_amount_results
        for f in glob.glob(self.temp_folder):
            try:
                os.unlink(f)
            except Exception as e:
                print(e)
                print("no file is observed!!!")

    def execute_clv(self, start_date, job='train_prediction', time_period='6 months'):
        """
        1. train clv prediction models
           For more details about models pls check; https://github.com/caglanakpinar/clv_prediction
        2. predict users of feature payment amount via using built models.
        3. use 'train_prediction' for the job argument which allows us to train and predict the results.
           If time period is weekly and if there is a stored .json and .h5 trained models in last 7 days,
           skips the train process and uses the latest trained model.
        4. Each model and stored in the directory at elasticsearch folder. After it is used, it is removed.
        5. If last stored clv prediction is applied before 7 days, it is not triggered for prediction again.

        :param start_date: date of start for clv prediction
        :param job: train or prediction
        :param time_period: weekly, daily, monthly
        """
        start_date = str(self.current_date_to_day())[0:10] if start_date is None else start_date
        self.check_for_previous_clv_prediction()
        if not self.has_prev_clv:
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
            self.clv_predictions = self.clv.get_result_data().query(self.result_query)[self.result_columns]
            self.clv_predictions = self.clv_predictions.rename(columns={"session_start_date": "date"})
            self.clv_predictions['date'] = self.clv_predictions['date'].apply(self.convert_to_iso_format)
            self.add_dimensions()
            self.clv_predictions = self.clv_predictions[['dimension', 'date', 'payment_amount', 'client']]
            self.insert_into_reports_index(
                ml_name="clv_prediction",
                ml=self.clv_predictions,
                                           time_period=time_period,
                                           date=start_date,
                                           index=self.order_index)
            del self.clv
            del self.clv_predictions
            self.clv = None
            self.clv_predictions = pd.DataFrame()
            self.remove_temp_files()

    def add_dimensions(self):
        _prediction_start_date = min(self.clv_predictions['date'])
        if len(self.client_dimensions) != 0:
            _clv_predictions_newcomers = self.clv_predictions.query("client == 'newcomers'")
            print(_clv_predictions_newcomers.head())
            self.clv_predictions = pd.merge(self.clv_predictions.query("client != 'newcomers'"),
                                            self.client_dimensions[['client', 'dimension']], on="client", how="left")
            # add newcomers of dimensions
            _dimension_user_ratios = self.clv_predictions.groupby('dimension').agg(
                {'client': lambda x: len(np.unique(x))}).reset_index().rename(columns={'client': 'ratio'})
            _dimension_user_ratios['ratio'] = _dimension_user_ratios['ratio'] / len(
                self.clv_predictions['client'].unique())
            _clv_predictions_newcomers = _clv_predictions_newcomers.groupby("date").agg(
                {"payment_amount": "sum"}).reset_index().rename(columns={'payment_amount': 'pa'})
            _clv_predictions_newcomers['client'] = 'newcomers'
            dfs = []
            for _ratio in list(_dimension_user_ratios['ratio']):
                _clv_predictions_newcomers['payment_amount'] = _clv_predictions_newcomers['pa'] * _ratio
                dfs.append(_clv_predictions_newcomers)
            self.clv_predictions = pd.concat([self.clv_predictions.query("client != 'newcomers'")] + dfs)
        else: self.clv_predictions['dimension'] = 'main'
        self.clv_predictions['dimension'] = self.clv_predictions['dimension'].apply(lambda x: str(x))
        self.clv_predictions = self.clv_predictions.query("date > @_prediction_start_date")
