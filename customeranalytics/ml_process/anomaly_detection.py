from os.path import join, abspath
import pandas as pd
import numpy as np
from math import sqrt
import datetime
from scipy import stats
import sys

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop

from customeranalytics.configs import default_es_port, default_es_host
from customeranalytics.data_storage_configurations.query_es import QueryES
from customeranalytics.data_storage_configurations.reports import Reports
from customeranalytics.utils import current_date_to_day, convert_to_date, calculate_time_diff, \
    get_index_group, convert_to_iso_format, convert_to_day


class Anomaly:
    """
    dimensional Anomaly Detection;

    By using reports such as Cohorts, RFM, Clv Prediction, e.g., Abnormal values are able to be detected.

    Daily Funnel:
        Daily Retention is calculated per each action per day. Next, Anomaly Score is calculated by AutoEncoder.
        Then, outlier days are detected by anomaly scores

    Daily Cohort From 1 to 2, 2 to 3, 3 to 4:
        Daily Order Cohorts of Anomaly Scores are calculated by AutoEncoder with first 7 orders.

    Daily Cohort From Download to 1:
        Daily Download Cohorts of Anomaly Scores are calculated by AutoEncoder with first 7 orders.

    Daily Orders
        Daily Orders of anomalies are calculated

    CLV Prediction and Segmentation of Monetary and Frequency Anomaly:
         By using CLV predictions, frequency and monetary are calculated per customer.
         Difference between historic frequency and monetary and prediction of them are calculated.
         Then, difference of anomalies are assigned as 'decrease', 'increase' and 'Normal'.

    """
    def __init__(self,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        ******* ******** *****
        Dimensional :
        Anomaly Detection must be created individually for dimensions.
        For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimension can be assigned in order to create the Anomaly Detection.

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
        self.report_execute = Reports()
        self.daily_funnel = pd.DataFrame()
        self.cohorts = pd.DataFrame()
        self.cohorts_d = pd.DataFrame()
        self.daily_orders = pd.DataFrame()
        self.daily_orders_comparison = pd.DataFrame()
        self.clv_prediction = pd.DataFrame()
        self.rfm = pd.DataFrame()
        self.actions_sequence = pd.DataFrame()
        self.actions_sequence_1,  self.actions_sequence_2 = [], []
        self.feature_cohort = [str(i) for i in list(range(7))]
        self.p_funnel = {"epochs":  40, "batch_size":  256, "h_layers":  4, "encode_dim": 32, "lr":  0.01, 'activation': 'relu'}
        self.p_cohort = {"epochs": 200, "batch_size": 40, "h_layers": 4, "encode_dim": 32, "lr": 0.001, 'activation': 'relu'}
        self.features = []
        self.query_es = QueryES(port=port, host=host)
        self.reports = pd.DataFrame()

    def collect_clv(self):
        """
        clv prediction results are not stored with dimensions.
        Need to query individually.
        """
        clv = self.report_execute.collect_reports(self.port, self.host, 'main',
                                                        query={"report_name": "clv_prediction"})
        clv['report_date'] = clv['report_date'].apply(lambda x: convert_to_date(x))

        clv = pd.DataFrame(list(clv.sort_values(['report_name', 'report_date'], ascending=False)['data'])[0])
        if get_index_group(self.order_index) != 'main':
            clv = clv[clv['dimension'] == get_index_group(self.order_index)]
        if len(clv) != 0:
            self.clv_prediction = clv.rename(columns={"date": "session_start_date"})

    def get_reports(self, date=None):
        """
        collecting all reports from reports index.
        """
        date = current_date_to_day() if date is None else convert_to_date(date)
        end_date = date.isoformat()
        self.reports = self.report_execute.collect_reports(self.port,
                                                           self.host,
                                                           get_index_group(self.order_index),
                                                           query={'index': get_index_group(self.order_index),
                                                                  'end': end_date})
        self.reports['report_date'] = self.reports['report_date'].apply(lambda x: convert_to_date(x))
        self.reports = self.reports.sort_values(['report_name', 'report_date'], ascending=False)
        self.collect_clv()

    def detect_outlier(self, value, _mean, _var, _sample_size, left_tail=False):
        """
        Statistically calculation via 0.05 error rate.
        :param value: float / intiger value
        :param _mean:  average value of sample
        :param _var: variance of sample
        :param _sample_size: number of data point
        :param left_tail: is for left part outlier or right tail
        :return: True / False
        """
        accept = 1 if value > _mean + (2.58 * (sqrt(_var) / sqrt(_sample_size))) else 0
        if left_tail:
            accept = 1 if value < _mean - (2.58 * (sqrt(_var) / sqrt(_sample_size))) else 0
        return accept

    def calculating_loss_function(self, X, model_ae, features):
        """
        Lost values from AutoEncoder
        :param X: values list shape(feature count, number of sample size)
        :param model_ae: trained model
        :param features: feature list
        :return: anomaly scores
        """
        y_pred, Y = model_ae.predict(X), X
        anomaly_calculations = list(map(lambda x: np.mean([abs(x[0][f] - x[1][f]) for f in range(len(features))]),
                                        zip(y_pred, Y)))
        return anomaly_calculations

    def calculate_confident_intervals(self, mean, var, n, alpha):
        """
        Statistical confidence Interval in order to detect outliers
        :param mean: average of values
        :param var: variance of values
        :param n: sample size
        :param alpha: error rate; 0.01, 0.05 e.g.
        :return: list of left tail and right tail point
        """
        df = n - 1  # degree of freedom for two sample t - set
        cv = stats.t.ppf(1 - (alpha / 2), df)
        standart_error = cv * sqrt(var / n)
        confidence_intervals = [mean - standart_error, mean + standart_error]
        return confidence_intervals

    def significant_dec_inc_detection(self, perc, confidence_interval):
        """
        Decision of ourlier which it is detected as left side outlier or right si.de outlier.
        :param perc: value between 0 - 1
        :param confidence_interval: intervalu values [left_tail, right_tail]
        :return: 'nomal decrease/increase', 'significant decrease', 'significant increase'
        """
        result = 'nomal decrease/increase'
        if perc < confidence_interval[0]:
            result = 'significant decrease'
        if perc > confidence_interval[1]:
            result = 'significant increase'
        return result

    def build_model(self, X, features, params):
        """
        AutoEncoder Model Creation.
        :param X: feature set array
        :param features: feature set
        :param params: tuned parameters
        :return: trained model
        """
        _input = Input(shape=(len(features),))
        encoder1 = Dense(128, activation=params['activation'])(_input)
        encoder2 = Dense(64, activation=params['activation'])(encoder1)
        encoder3 = Dense(32, activation=params['activation'])(encoder2)
        encoder4 = Dense(16, activation=params['activation'])(encoder3)
        encoder5 = Dense(8, activation=params['activation'])(encoder4)
        code = Dense(len(features), activation=params['activation'])(encoder5)
        decoder1 = Dense(1, activation=params['activation'])(code)
        decoder2 = Dense(8, activation=params['activation'])(decoder1)
        decoder3 = Dense(16, activation=params['activation'])(decoder2)
        decoder4 = Dense(32, activation=params['activation'])(decoder3)
        decoder5 = Dense(64, activation=params['activation'])(decoder4)
        decoder6 = Dense(128, activation=params['activation'])(decoder5)
        fr_output = Dense(len(features), activation=params['activation'])(decoder6)
        model_ae = Model(inputs=_input, outputs=fr_output)
        model_ae.compile(loss='mse', optimizer=RMSprop(lr=params['lr']), metrics=['mse'])

        model_ae.fit(X, X,
                     epochs=int(params['epochs']),
                     batch_size=int(params['batch_size']),
                     validation_split=0.2, shuffle=True)
        anomaly_scores = self.calculating_loss_function(X, model_ae, features)
        return anomaly_scores

    def get_daily_funnel_anomaly(self):
        """
        Daily Funnel:
            Daily Retention is calculated per each action per day. Next, Anomaly Score is calculated by AutoEncoder.
            Then, outlier days are detected by anomaly scores
            - collect daily funnel from reports index. Type of funnel is 'orders'
            - find the actions of flow by using mean of their retention.
              The lowest numbers of sessions will be 'has_purchased' action.
              The highest numbers of sessions will be 'has_seesion' action.
            - by using the actions of flow, calculate retentions per each action per day
            - by using the actions of flow, calculate retention per day for 'has_sessions' - action_1,
              'has_sessions' - action_2, ... , 'has_sessions' - 'has_purchased',
            - User retention values as feature.
            - build AutoEncoder model.
            - calculate scores with trained model
            -


        :return:
        """
        self.daily_funnel = pd.DataFrame(list(self.reports.query(
            "report_name == 'funnel' and time_period == 'daily' and type == 'orders'")['data'])[0])
        self.actions_sequence = pd.DataFrame([{'action': i, 'mean': np.mean(self.daily_funnel[i])} for i in
                                         list(set(list(self.daily_funnel.columns)) - set(['daily']))])
        self.actions_sequence = self.actions_sequence.sort_values(by='mean', ascending=False)
        self.actions_sequence = list(self.actions_sequence['action'])

        self.actions_sequence_1 = list(zip([None] + self.actions_sequence, self.actions_sequence + [None]))

        for a in self.actions_sequence_1[1:-1]:
            self.daily_funnel['retantion_' + "_".join([a[0], a[1]])] = self.daily_funnel.apply(
                lambda row: row[a[1]] / row[a[0]], axis=1)

        if len(self.actions_sequence_2) > 2:
            for a in self.actions_sequence_2:
                self.daily_funnel['retantion_' + "_".join([a[0], a[1]])] = self.daily_funnel.apply(
                    lambda row: row[a[1]] / row[a[0]], axis=1)

        self.features = list(set(self.daily_funnel.columns) - set(['day', 'daily', 'weekday', 'daily_purchased',
                                                                   'daily_has_sessions', 'daily_has_basket',
                                                                   'daily_order_screen']))
        self.daily_funnel['anomaly_scores'] = self.build_model(self.daily_funnel[self.features].values,
                                                               self.features, self.p_funnel)
        _mean, _var, _sample_size = np.mean(self.daily_funnel['anomaly_scores']), np.var(
            self.daily_funnel['anomaly_scores']), len(self.daily_funnel)
        self.daily_funnel['outlier'] = self.daily_funnel['anomaly_scores'].apply(
            lambda x: self.detect_outlier(x, _mean, _var, _sample_size))
        self.daily_funnel = self.daily_funnel[['outlier', 'anomaly_scores', 'daily']]

    def get_cohort_anomaly(self):
        _features = []
        for o in [2, 3, 4]:
            print("********* number of orders :", o)
            print("report_name == 'cohort' and time_period == 'daily' and _to == " + str(o) + " ")
            _cohort = pd.DataFrame(list(self.reports.query(
                "report_name == 'cohort' and time_period == 'daily' and _to == " + str(o) + " ")['data'])[0])
            _name = 'anomaly_scores_' + str(o)
            _cohort[_name] = self.build_model(_cohort[self.feature_cohort].values,
                                              self.feature_cohort, self.p_cohort)
            if len(self.cohorts) == 0:
                self.cohorts = _cohort[['daily', _name]]
            else:
                self.cohorts = pd.merge(self.cohorts, _cohort[['daily', _name]], on='daily', how='left')
            _features.append(_name)
        self.cohorts['anomaly_score'] = self.build_model(self.cohorts[_features].values, _features, self.p_cohort)
        _mean, _var, _sample_size = np.mean(self.cohorts['anomaly_score']), np.var(self.cohorts['anomaly_score']), len(self.cohorts)
        self.cohorts['outlier'] = self.cohorts['anomaly_score'].apply(
            lambda x: self.detect_outlier(x, _mean, _var, _sample_size))
        self.cohorts = self.cohorts[['outlier', 'anomaly_score', 'daily']]

    def get_download_cohort_anomaly(self):
        self.cohorts_d = pd.DataFrame(
            list(self.reports.query(
                "report_name == 'cohort' and time_period == 'daily' and type == 'downloads'")['data'])[0])
        self.features = [str(i) for i in list(range(7))]
        self.cohorts_d['anomaly_scores_from_d_to_1'] = self.build_model(self.cohorts_d[self.features].values,
                                                                        self.features, self.p_cohort)
        _mean, _var, _sample_size = np.mean(self.cohorts_d['anomaly_scores_from_d_to_1']), np.var(
            self.cohorts_d['anomaly_scores_from_d_to_1']), len(self.cohorts_d)
        self.cohorts_d['outlier'] = self.cohorts_d['anomaly_scores_from_d_to_1'].apply(
            lambda x: self.detect_outlier(x, _mean, _var, _sample_size, left_tail=True))
        self.cohorts = self.cohorts[['outlier', 'anomaly_score', 'daily']]

    def get_daily_orders_anomaly(self):
        self.daily_orders = pd.DataFrame(list(self.reports.query(
            "report_name == 'stats' and type == 'daily_orders'")['data'])[0])
        self.daily_orders['daily'] = self.daily_orders['daily'].apply(lambda x: convert_to_date(x))
        self.daily_orders['isoweekday'] = self.daily_orders['daily'].apply(lambda x: x.isoweekday())
        max_isoweekday = max(self.daily_orders['daily']).isoweekday()
        number_of_week_back_iteration = len(self.daily_orders[self.daily_orders['isoweekday'] == max_isoweekday]) - 5
        _data = self.daily_orders
        for i in range(1, number_of_week_back_iteration + 1):
            for _diff in [(1, 'is_last_order'), (5, 'is_last_month')]:
                _data[_diff[1]] = False
                _data[_diff[1]] = _data.sort_values(["daily", "isoweekday"], ascending=True).groupby("isoweekday")[
                    _diff[1]].shift(-_diff[0])
                _data[_diff[1]] = _data[_diff[1]].fillna(True)
            _last_month_recent_day_compare = _data.query("is_last_month == True").groupby(
                ["is_last_order", "isoweekday"]).agg({"orders": "mean"}).reset_index()

            _last_month_recent_day_compare = pd.merge(
                _last_month_recent_day_compare.query(
                    "is_last_order == False").rename(columns={"orders": "order_count_last_month"}),
                _last_month_recent_day_compare.query(
                    "is_last_order == True").rename(columns={"orders": "order_count_last_recent"}),
                on='isoweekday', how='inner')[['isoweekday', 'order_count_last_month', 'order_count_last_recent']]
            _last_month_recent_day_compare['diff_perc'] = 100 * (
                    _last_month_recent_day_compare['order_count_last_recent'] - _last_month_recent_day_compare[
                'order_count_last_month']) / _last_month_recent_day_compare['order_count_last_month']
            self.daily_orders_comparison = pd.concat([
                self.daily_orders_comparison,
                pd.merge(_data.query("is_last_order == True")[['daily', 'isoweekday']],
                         _last_month_recent_day_compare, on='isoweekday', how='left')])
            _data = _data.query("is_last_order != True")

        confidence_interval = self.calculate_confident_intervals(np.mean(self.daily_orders_comparison['diff_perc']),
                                                                 np.var(self.daily_orders_comparison['diff_perc']),
                                                                 len(self.daily_orders_comparison), 0.05)

        self.daily_orders_comparison['anomalities'] = self.daily_orders_comparison['diff_perc'].apply(
            lambda x: self.significant_dec_inc_detection(x, confidence_interval))
        self.daily_orders_comparison = self.daily_orders_comparison[['diff_perc', 'anomalities', 'daily']]

    def clv_segmentation_change(self):
        """
        Each client of monetary and frequency change related to CLV Predictions.
        On CLV Prediction customers of future expected order date and purchase_amount values can be detected.
        """
        self.collect_clv()
        self.rfm = pd.DataFrame(list(self.reports.query("report_name == 'rfm'")['data'])[0])
        self.clv_prediction['session_start_date'] = self.clv_prediction['session_start_date'].apply(
            lambda x: convert_to_day(x))
        self.clv_prediction['session_start_date_prev'] = self.clv_prediction.sort_values(
            by=['client', 'session_start_date']).groupby(['client'])['session_start_date'].shift(-1)

        frequency_clv = self.clv_prediction[['session_start_date_prev', 'session_start_date', 'client']]
        frequency_clv['session_start_date_prev'] = frequency_clv['session_start_date_prev'].fillna(current_date_to_day())
        frequency_clv['frequency'] = frequency_clv.apply(
            lambda row: calculate_time_diff(row['session_start_date'], row['session_start_date_prev'], 'week'), axis=1)
        frequency_clv = frequency_clv.groupby('client').agg({'frequency': 'mean'}).reset_index()

        monetary_clv = self.clv_prediction.groupby('client').agg({"payment_amount": "mean"}).reset_index()

        self.clv_prediction = pd.merge(pd.merge(monetary_clv,
                                                frequency_clv, on='client', how='left'),
                                       self.rfm.rename(columns={'frequency': 'frequency_prev',
                                                                'monetary': 'monetary_prev'}),
                                       on='client', how='left')

        self.clv_prediction['monetary_diff'] = self.clv_prediction['payment_amount'] - self.clv_prediction['monetary_prev']
        self.clv_prediction['frequency_diff'] = self.clv_prediction.apply(
            lambda row: row['frequency_prev'] - row['frequency'] if row['frequency'] == row['frequency'] else None,
            axis=1)
        avg_frequency_diff = np.mean(self.clv_prediction.query("frequency_diff == frequency_diff")['frequency_diff'])
        self.clv_prediction['frequency_diff'] = self.clv_prediction['frequency_diff'].fillna(avg_frequency_diff)

        self.clv_prediction['m_dec_inc'] = self.clv_prediction['monetary_diff'].apply(
            lambda x: 'decrease' if x < 0 else 'increase')
        self.clv_prediction['f_dec_inc'] = self.clv_prediction['frequency_diff'].apply(
            lambda x: 'decrease' if x < 0 else 'increase')

        m_confidence_interval = self.calculate_confident_intervals(np.mean(self.clv_prediction['monetary_diff']),
                                                                   np.var(self.clv_prediction['monetary_diff']),
                                                                   len(self.clv_prediction), 0.05)

        f_confidence_interval = self.calculate_confident_intervals(np.mean(self.clv_prediction['frequency_diff']),
                                                                   np.var(self.clv_prediction['frequency_diff']),
                                                                   len(self.clv_prediction), 0.05)

        self.clv_prediction['m_anomaly'] = self.clv_prediction['monetary_diff'].apply(
            lambda x: self.significant_dec_inc_detection(x, m_confidence_interval))
        self.clv_prediction['f_anomaly'] = self.clv_prediction['frequency_diff'].apply(
            lambda x: self.significant_dec_inc_detection(x, f_confidence_interval))
        self.clv_prediction = self.clv_prediction[['f_anomaly', 'm_anomaly',
                                                   'monetary_diff', 'frequency_diff', 'f_dec_inc', 'm_dec_inc']]

    def execute_anomaly(self, date):
        self.get_reports(date)
        self.get_daily_funnel_anomaly()
        self.get_cohort_anomaly()
        self.get_download_cohort_anomaly()
        self.get_daily_orders_anomaly()
        self.clv_segmentation_change()
        dfs = [self.daily_funnel, self.cohorts, self.cohorts_d, self.daily_orders_comparison, self.clv_prediction]
        names = ['daily_funnel', 'cohort', 'cohort_d', 'daily_orders_comparison', 'clv_prediction']
        for i in zip(names, dfs):
            print("names :", i[0])
            self.insert_into_reports_index(i[0], i[1], current_date_to_day(), index=self.order_index)

    def insert_into_reports_index(self, name, anomaly, start_date, index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": start_date or current date,
         "report_name": "rfm",
         "index": "main",
         "report_types": {},
         "data": rfm.fillna(0.0).to_dict("results") -  dataframe to list of dictionary
         }
         !!! null values are assigned to 0.

        :param rfm: data set, data frame
        :param start_date: data start date
        :param index: dimensionality of data index orders_location1 ;  dimension = location1
        """

        list_of_obj = [{"id": np.random.randint(200000000),
                        "report_date": current_date_to_day().isoformat() if start_date is None else start_date,
                        "report_name": "anomaly",
                        "index": get_index_group(index),
                        "report_types": {"type": name},
                        "data": anomaly.fillna(0).to_dict("results")}]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def fetch(self, anomly, start_date=None):
        """
        Collect RFM values for each user. Collecting stored RFM is useful in order to initialize Customer Segmentation.
        :return: data-frame
        """

        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_name": "anomaly"}},
                           {"term": {"index": get_index_group(self.order_index)}},
                           {"term": {"report_types.type": anomly}}
                           ]

        if start_date is not None:
            date_queries = [{"range": {"report_date": {"gte": convert_to_iso_format(start_date)}}}]

        self.query_es = QueryES(port=self.port,
                                host=self.host)
        self.query_es.query_builder(fields=None, _source=True,
                                    date_queries=date_queries,
                                    boolean_queries=boolean_queries)
        _res = self.query_es.get_data_from_es(index="reports")
        _data = pd.DataFrame()
        if len(_res) != 0:
            _data = pd.DataFrame(_res[0]['_source']['data'])
        return _data