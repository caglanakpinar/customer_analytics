from os.path import join, abspath
import pandas as pd
import numpy as np
from math import sqrt
import datetime
from scipy import stats
import pygeohash as gh
import random
import shutil
import time

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

from customeranalytics.data_storage_configurations.query_es import QueryES
from customeranalytics.configs import not_required_default_values, default_es_port, default_es_host, \
    delivery_anomaly_model_parameters, delivery_anomaly_model_hyper_parameters
from customeranalytics.utils import convert_to_date, get_index_group, dimension_decision, \
    current_date_to_day, find_week_of_monday


class DeliveryAnalytics:
    """
    Delivery Analytics;

    These analysis are only able to be run when delivery data source is created.
    It detects abo-normal durations of rides according to customer location, location of order, hour of the order,
    weekday of the order.

    """
    def __init__(self,
                 temporary_export_path,
                 has_delivery_connection=True,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        ******* ******** *****
        Dimensional :
        Delivery Analytics must be created individually for dimensions.
        For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimension can be assigned in order to create the Delivery Analytics.

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
        :param has_delivery_connection: is any additional delivery data source
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.download_index = download_index
        self.order_index = order_index
        self.has_delivery_connection = has_delivery_connection
        self.temporary_export_path = temporary_export_path
        self.query_es = QueryES(port=self.port, host=self.host)
        self.data = pd.DataFrame()
        self.has_data_end_date = False
        self.has_location_data = False
        self.features = lambda x: list(range(1, 8)) if x == 'weekday_hour' else list(range(1, 6))
        self.features_norm = lambda x: [str(i) + '_norm' for i in self.features(x)]
        self.model_data = {}
        self.centroid = []
        self.duration_metrics = ['deliver', 'prepare', 'ride', 'returns']
        self.anomaly_metrics = ["customer", 'location', 'weekday_hour']
        self._model = {}
        self.params = delivery_anomaly_model_parameters
        self.hyper_params = delivery_anomaly_model_hyper_parameters
        self.hp = HyperParameters()
        self.parameter_tuning_trials = 5  # number of parameter tuning trials
        self.delivery_fields = ['id', 'client', 'session_start_date', 'date', 'payment_amount', 'discount_amount',
                                'delivery.delivery_date', 'delivery.prepare_date',
                                'delivery.return_date', 'delivery.latitude', 'delivery.longitude']
        self.model = {m1: {m2: {"data": pd.DataFrame(),
                                "feature_data": pd.DataFrame(),
                                "train": [],
                                "test": [],
                                'features': self.features(m1),
                                'features_norm': self.features_norm(m1),
                                'model': None,
                                'params': self.params,
                                'hyper_params': self.hyper_params}
                           for m2 in self.anomaly_metrics}
                      for m1 in self.duration_metrics}
        self.functions = {m[0]: m[1] for m in
            zip(self.anomaly_metrics, [self.customer_delivery_anomaly,
                                       self.location_delivery_anomaly, self.weekday_hour_delivery_anomaly])}

    def dimensional_query(self, boolean_query=None):
        if dimension_decision(self.order_index):
            if boolean_query is None:
                boolean_query = [{"term": {"dimension": self.order_index}}]
            else:
                boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def get_delivery_data(self):
        """
        collect data for delivery anomaly from orders index.
        It needs only purchased orders.
        Delivery object needs to be split.
        """
        self.query_es.query_builder(fields=self.delivery_fields,
                                    boolean_queries=self.dimensional_query([{"term": {"actions.purchased": True}}]))

        self.data = pd.DataFrame(self.query_es.get_data_from_es()).rename(
            columns={'id': 'order_id', 'delivery.delivery_date': 'delivery_date',
                     'delivery.prepare_date': 'prepare_date', 'delivery.return_date': 'return_date',
                     'delivery.latitude': 'latitude', 'delivery.longitude': 'longitude'})
        
    def data_source_date_decision(self, row, dt):
        """
        Check each columns 'date', 'prepare_date', 'return_date', 'latitude', 'longitude'
        that it is assigned as default value of them

        :param row: each row at data-frame
        :param dt:  columns names 'date', 'prepare_date', 'return_date', 'latitude', 'longitude'
        """
        if row[dt] == not_required_default_values.get(dt, None):
            # get(dt, None )this is only for end_date (date) columns
            return True
        else:
            return False

    def get_has_data_end_date(self):
        """
        check data that has session end date.
        Count unique end_date values and count total number of purchased sessions and calcualte the ratio
        if the ratio is more than .1, end_date is assigned with sessions data source
        """
        end_date = list(self.data['date'])
        unique_end_date = set(end_date)
        if len(unique_end_date) / len(end_date) > 0.1:
            self.has_data_end_date = True

    def delivery_durations(self, row):
        """
        *** Duration Caşculations ***

            prepare duration;
                If prepare date is assigned it is possible to be calculated.
                prepare = end_date - prepare_date
            delivery duration;
                If delivery data source is assigned delivery_date is required so, delivery durations can be calculated.
                delivery = prepare_date - delivery_date
            return duration;
                If return date is assigned return duration can be calculated.
                return = delivery_date - return_date

        :param row: each row at data-frame
        """
        deliver, prepare, ride, returns = ['-'] * 4
        if self.has_data_end_date:  # if there is no significant end_date, assigned as session_start_date
            session_end = row['date']
        else:
            session_end = row['session_start_date']

        # if there is no prepare_date assigned it as end_date.
        # It end_date is also not existed assigned as session_start_date
        if row['prepare_date'] is not None:
            prepare_date = row['prepare_date']
            ride_start_date = row['prepare_date']
        else:
            if self.has_data_end_date:
                prepare_date = row['date']
                ride_start_date = row['prepare_date']
            else:
                prepare_date = None
                ride_start_date = row['session_start_date']

        # If there is no return_date, assign it as None
        if row['return_date'] is not None:
            return_date = row['return_date']
        else:
            return_date = None

        # delivery_date is required column for delivery dta source
        deliver = abs(row['delivery_date'] - session_end).total_seconds() / 60
        # if both prepare_date and end_date is not assigned, both will be assigned as session_start_date.
        # So, If they are not assigned prepare duration will be 0
        if prepare_date == prepare_date:
            prepare = abs(prepare_date - session_end).total_seconds() / 60
        ride = abs(row['delivery_date'] - ride_start_date).total_seconds() / 60
        # return duration will be assigned as '-' if duration_date isn`t existed
        if return_date == return_date:
            returns = abs(row['return_date'] - row['delivery_date']).total_seconds() / 60
        return pd.Series([deliver, prepare, ride, returns])

    def encode_geo_hash(self, row):
        try:
            return gh.encode(row['latitude'], row['longitude'], precision=8)
        except Exception as e:
            return None

    def parse_lat_lon(self, geohash):
        return pd.Series([geohash[0], geohash[1]])

    def geo_hash_perc7(self, _location, perc):
        _location['location_perc%s' % str(perc)] = _location.apply(
            lambda row: gh.decode(row['geohash_perc%s' % str(perc)]), axis=1)
        _location[['lat%s' % str(perc), 'lon%s' % str(perc)]] = _location.apply(
            lambda row: self.parse_lat_lon(row['location_perc%s' % str(perc)]), axis=1)
        return _location

    def convert_data_format(self, value, data_type):
        if data_type in ['latitude', 'longitude']:
            return float(value)
        else:
            return convert_to_date(value)

    def data_manipulations(self):
        for dt in ['session_start_date', 'delivery_date']:
            self.data[dt] = self.data.apply(lambda row: convert_to_date(row[dt]), axis=1)

        for dt in ['date', 'prepare_date', 'return_date', 'latitude', 'longitude']:
            self.data[dt] = self.data.apply(
                lambda row: self.convert_data_format(row[dt], dt)
                if not self.data_source_date_decision(row, dt) else None, axis=1)

        for a in ['payment_amount', 'discount_amount']:
            self.data[a] = self.data[a].apply(lambda x: float(x))

        # geo hash encoding-decoding precision=7
        self.data['geohash_perc7'] = self.data.apply(lambda row: self.encode_geo_hash(row), axis=1)

        # weekday and hour
        self.data['isoweekday'] = self.data['session_start_date'].apply(lambda x: x.isoweekday())
        self.data['hour'] = self.data['session_start_date'].apply(lambda x: x.hour)
        self.data['week'] = self.data['session_start_date'].apply(lambda x: find_week_of_monday(x))

        # duration calculations; ride, delivery, prepare
        self.data[self.duration_metrics] = self.data.apply(lambda row: self.delivery_durations(row), axis=1)

    def min_max_norm(self, value, _min, _max):
        if abs(_max - _min) != 0:
            return (value - _min) / abs(_max - _min)
        else:
            return 0

    def min_max_norm_reverse(self, norm_value, _min, _max):
        return (norm_value * abs(_max - _min)) + _min

    def create_feature_set_and_train_test_split(self):
        self._model['features_norm'] = [str(i) + '_norm' for i in range(1, 6)]
        self._model['feature_data'] = self._model['feature_data'].reset_index(drop=True).reset_index()
        index = list(self._model['feature_data']['index'])
        _train_size = int(len(index) * 0.8)
        _test_size = len(index) - _train_size
        _train_index = random.sample(index, _train_size)
        _test_index = list(set(index) - set(_train_index))

        self._model['train'] = self._model['feature_data'].query("index in @_train_index")[self._model['features_norm']].values
        self._model['test'] = self._model['feature_data'].query("index in @_test_index")[self._model['features_norm']].values

    def build_parameter_tuning_model(self, hp):
        _input = Input(shape=(self._model['train'].shape[1],))
        _unit = hp.Choice('h_l_unit', self._model['hyper_params']['h_l_unit'])
        _layer = Dense(hp.Choice('h_l_unit', self._model['hyper_params']['h_l_unit']),
                       activation=hp.Choice('activation', self._model['hyper_params']['activation'])
                       )(_input)

        for i in range(1, hp.Choice('hidden_layer_count', self._model['hyper_params']['hidden_layer_count'])):
            _unit = _unit / 2
            _layer = Dense(_unit,
                           activation=hp.Choice('activation', self._model['hyper_params']['activation'])
                           )(_layer)

        output = Dense(self._model['train'].shape[1], activation='sigmoid')(_layer)
        model = Model(inputs=_input, outputs=output)
        model.compile(loss=self._model['hyper_params']['loss'],
                      optimizer=Adam(lr=hp.Choice('lr', self._model['hyper_params']['lr'])))
        return model

    def remove_keras_tuner_folder(self):
        """
        removing keras tuner file. while you need to update the parameters it will affect rerun the parameter tuning.
        It won`t start unless the folder has been removed.
        """

        try:
            shutil.rmtree(join(self.temporary_export_path, "delivery_anomaly"))
        except Exception as e:
            print(" Parameter Tuning Keras Turner dummy files have already removed!!")

    def parameter_tuning(self):
        kwargs = {'directory': join(self.temporary_export_path, "delivery_anomaly")}
        tuner = RandomSearch(self.build_parameter_tuning_model,
                             max_trials=self.parameter_tuning_trials,
                             hyperparameters=self.hp,
                             allow_new_entries=True,
                             objective='loss', **kwargs)
        tuner.search(x=self._model['train'],
                     y=self._model['train'],
                     epochs=5,
                     batch_size=self._model['hyper_params']['batch_size'],
                     verbose=1,
                     validation_data=(self._model['test'], self._model['test']))

        for p in tuner.get_best_hyperparameters()[0].values:
            if p in list(self._model['params'].keys()):
                self._model['params'][p] = tuner.get_best_hyperparameters()[0].values[p]
        self.remove_keras_tuner_folder()

    def calculating_loss_function(self, X, model, features):
        """
        Lost values from AutoEncoder
        :param X: values list shape(feature count, number of sample size)
        :param model: trained model
        :param features: feature list
        :return: anomaly scores
        """
        y_pred, Y = model.predict(X), X
        anomaly_calculations = list(map(lambda x: np.mean([abs(x[0][f] - x[1][f]) for f in range(len(features))]),
                                        zip(y_pred, Y)))
        return anomaly_calculations

    def norm_values_outlier(self, scores):
        mean_scores = np.mean(scores)
        std_scores = np.std(scores)
        standart_error = 2.58 * sqrt(std_scores / len(scores))
        return mean_scores + standart_error

    def detect_outliers(self, type):
        self._model['feature_data']['anomaly_scores'] = self.calculating_loss_function(
            self._model['feature_data'][self._model['features_norm']].values,
            self._model['model'], self._model['features_norm'])
        standart_error_right_tail = self.norm_values_outlier(list(self._model['feature_data']['anomaly_scores']))
        self._model['feature_data'][type + '_anomaly'] = self._model['feature_data']['anomaly_scores'].apply(
            lambda x: 1 if x > standart_error_right_tail else 0)

    def build_model(self, type):
        _input = Input(shape=(self._model['train'].shape[1],))
        _unit = self._model['params']['h_l_unit']
        _layer = Dense(self._model['params']['h_l_unit'],
                       activation=self._model['params']['activation']
                       )(_input)

        for i in range(1, self._model['params']['hidden_layer_count']):
            _unit = _unit / 2
            _layer = Dense(_unit, activation=self._model['params']['activation'])(_layer)

        output = Dense(self._model['train'].shape[1], activation='sigmoid')(_layer)
        self._model['model'] = Model(inputs=_input, outputs=output)
        self._model['model'].compile(loss='mse', optimizer=Adam(lr=self._model['params']['lr']), metrics=['mse'])
        self._model['model'].fit(self._model['train'], self._model['train'],
                                 epochs=int(self._model['params']['epochs']),
                                 batch_size=int(self._model['params']['batch_size']),
                                 verbose=True,
                                 validation_split=0.2, shuffle=True)

        self.detect_outliers(type)

    def calculate_features_min_max_normalization(self):
        max_value = self._model['feature_data'][self._model['features']].values.max()
        min_value = self._model['feature_data'][self._model['features']].values.min()
        for i in self._model['features']:
            self._model['feature_data'][str(i) + '_norm'] = self._model['feature_data'][i].apply(
                lambda x: self.min_max_norm(float(x), min_value, max_value))

    def customer_delivery_anomaly(self, metric):
        """

        """
        client_deliveries = self.data.groupby("client").agg(
            {metric: "mean", "order_id": "count"}).reset_index().sort_values(by='order_id', ascending=False)
        client_deliveries = client_deliveries.query("order_id > 5")
        clients = list(client_deliveries['client'].unique())
        self._model['data'] = self.data.query("client in @clients").sort_values(
            ['client', 'session_start_date'], ascending=True)
        self._model['data']['order_seq'] = self._model['data'].sort_values(by=['client', 'session_start_date']).groupby(
            ['client']).cumcount() + 1

        self._model['data'] = self._model['data'].merge(
            self._model['data'].groupby('client').agg({"order_seq": "max"}).reset_index().rename(
                columns={"order_seq": "order_seq_max"}),
            on='client', how='left')
        self._model['data']['order_seq'] = self._model['data']['order_seq_max'] - self._model['data']['order_seq']
        self._model['data'] = self._model['data'].query("order_seq_max > 5 and order_seq < 5")
        self._model['feature_data'] = pd.DataFrame(np.array(self._model['data'].pivot_table(columns='order_seq',
                                                                                index='client',
                                                                                aggfunc={metric: "mean"}
                                                                       ).reset_index())).rename(
            columns={0: "client"}).reset_index(drop=True).reset_index().fillna("-")

        for i in self._model['features']:
            self._model['feature_data'] = self._model['feature_data'][self._model['feature_data'][i] != '-']

        self.calculate_features_min_max_normalization()
        self.create_feature_set_and_train_test_split()
        self.parameter_tuning()
        self.build_model('customer')
        self.model[metric]['customer'] = self._model

    def location_delivery_anomaly(self, metric):
        """

        """
        self._model['data'] = self.data.groupby(["week", 'geohash_perc7']).agg({metric: "mean"}).reset_index()
        self._model['data']['week_seq'] = self._model['data'].sort_values(by=['geohash_perc7', 'week']).groupby(
            ['geohash_perc7']).cumcount() + 1

        locations_max_weeks = self._model['data'].groupby("geohash_perc7").agg(
            {"week_seq": "max"}).reset_index().rename(columns={"week_seq": "week_seq_max"})
        self._model['data'] = self._model['data'].merge(locations_max_weeks, on='geohash_perc7', how='left')
        self._model['data']['week_seq'] = self._model['data']['week_seq_max'] - self._model['data']['week_seq']

        self._model['feature_data'] = pd.DataFrame(np.array(self._model['data'].query("week_seq < 5").pivot_table(
            columns="week_seq",
            index='geohash_perc7',
            aggfunc={metric: "max"}).reset_index())).rename(columns={0: "location"})

        for f in self._model['features']:
            mean_duration = self._model['feature_data'][self._model['feature_data'][f] == self._model['feature_data'][f]]
        mean_duration = mean_duration[self._model['features']].values.mean()
        self._model['feature_data'] = self._model['feature_data'].fillna(mean_duration)

        self.calculate_features_min_max_normalization()
        self.create_feature_set_and_train_test_split()
        self.parameter_tuning()
        self.build_model('location')
        self.model[metric]['location'] = self._model

    def weekday_hour_delivery_anomaly(self, metric):
        """

        """
        self._model['feature_data'] = self.data.groupby(["isoweekday", "hour"]).agg(
            {metric: "mean"}).reset_index().rename(columns={metric: "weekday_hour_norm"})
        max_value = self._model['feature_data'][["weekday_hour_norm"]].values.max()
        min_value = self._model['feature_data'][["weekday_hour_norm"]].values.min()
        self._model['feature_data']['weekday_hour_norm'] = self._model['feature_data']['weekday_hour_norm'].apply(
            lambda x: self.min_max_norm(float(x), min_value, max_value))

        self._model['feature_data']['weekday_hour_anomaly'] = self._model['feature_data']['weekday_hour_norm'].apply(
            lambda x: 1 if x > 0.9 else 0)

    def merge_type_anomaly_data(self, data, metric):
        # customers of detected abnormal transactions
        data = data.merge(self.model[metric]['customer']['feature_data'][['client', 'customer_anomaly']],
                          on='client', how='left')

        # locations (geo-hashed (perc=7)) of detected abnormal transactions
        if self.has_location_data:
            data = data.merge(
                self.model[metric]['location']['feature_data'].rename(
                    columns={"location": "geohash_perc7"})[['geohash_perc7', 'location_anomaly']],
                on='geohash_perc7', how='left')
        else:
            data['location_anomaly'] = 0

        # weekday - hour of detected abnormal transactions
        data = data.merge(self.model[metric]['weekday_hour']['feature_data']
                          [['hour', 'isoweekday',  'weekday_hour_anomaly']], on=['hour', 'isoweekday'], how='left')

        data['anomaly_totals'] = data['customer_anomaly'] + data['location_anomaly'] + data['weekday_hour_anomaly']
        data = data.query("anomaly_totals == 3 or ((customer_anomaly == 1 and weekday_hour_anomaly == 1) or (location_anomaly == 1 and weekday_hour_anomaly == 1))")
        return data[['session_start_date', 'hour', 'isoweekday', 'latitude', 'longitude',
                     metric, 'client', 'customer_anomaly', 'location_anomaly', 'weekday_hour_anomaly']]

    def decision_for_location_anomaly(self):
        if len(list(self.data['latitude'].unique())) > 1 or len(list(self.data['latitude'].unique())) > 1:
            self.has_location_data = True
        else:
            self.anomaly_metrics = list(set(self.anomaly_metrics) - {'location'})

    def delivery_kpis(self):
        kpis = {}
        for m in self.duration_metrics:
            kpis[m] = np.mean(self.data[m])

        self.data['locations'] = self.data.apply(lambda row: "_".join([str(row['latitude']), str(row['longitude'])]), axis=1)
        kpis['total_locations'] = len(np.unique(self.data['locations']))
        return pd.DataFrame([kpis])

    def execute_delivery_analysis(self):
        """

        """
        if self.has_delivery_connection:
            self.get_delivery_data()
            self.get_has_data_end_date()
            self.data_manipulations()
            self.decision_for_location_anomaly()

            # anomaly calculations
            for metric in self.duration_metrics:
                for type in self.anomaly_metrics:
                    print("type :", type, " || metric :", metric)
                    self._model = self.model[metric][type]
                    self.functions[type](metric)
                _result = self.merge_type_anomaly_data(self.data, metric)
                self.insert_into_reports_index(delivery_anomaly=_result, anomaly_type=metric, index=self.order_index)

            # visualization and analytics calculation
            for metric in self.duration_metrics:
                self.insert_into_reports_index(delivery_anomaly=self.model[metric]['weekday_hour']['feature_data'],
                                               anomaly_type=metric + '_weekday_hour', index=self.order_index)

            if self.has_location_data:
                for metric in self.duration_metrics:
                    _location = self.model[metric]['location']['data']
                    _location = self.geo_hash_perc7(_location, perc=7)
                    if sum(_location[metric]) != 0:
                        self.insert_into_reports_index(delivery_anomaly=_location,
                                                       anomaly_type=metric + '_location', index=self.order_index)
            # delivery KPIs
            self.insert_into_reports_index(delivery_anomaly=self.delivery_kpis(),
                                           anomaly_type='deliver_kpis', index=self.order_index)

    def insert_into_reports_index(self, delivery_anomaly, start_date=None, anomaly_type='ride', index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": start_date or current date,
         "report_name": "delivery_anomaly",
         "index": "main",
         "report_types": {
                          "type": anomaly_type; 'deliver', 'prepare', 'ride', 'returns'
                          },
         "data": delivery_anomaly
         }
        :param delivery_anomaly: data set, data frame
        :param start_date: data start date
        :param anomaly_type: data types; 'deliver', 'prepare', 'ride', 'returns'
        :param index: dimentionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = [{"id": np.random.randint(200000000),
                        "report_date": current_date_to_day().isoformat() if start_date is None else start_date,
                        "report_name": "delivery_anomaly",
                        "index": get_index_group(index),
                        "report_types": {"type": anomaly_type},
                        "data": delivery_anomaly.fillna(0).to_dict("results")}]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def fetch(self, anomaly_type, start_date=None):
        """
        This allows us to query the created delivery_anomaly reports.
        anomaly_type is crucial for us to collect the correct filters.
        Example of queries;
            -   anomaly_type: funnel_downloads_daily,
            -   start_date: 2021-01-01T00:00:00

            {'size': 10000000,
            'from': 0,
            '_source': True,
            'query': {'bool': {'must': [
                                        {'term': {'report_name': 'delivery_anomaly'}},
                                        {"term": {"index": "orders_location1"}}
                                        {'term': {'report_types.type': 'ride'}},
                                        {'range': {'report_date': {'lt': '2021-04-01T00:00:00'}}}]}}}

            - start date will be filtered from data frame. In this example; .query("daily > @start_date")

        :param anomaly_type: 'deliver', 'prepare', 'ride', 'returns'
        :param start_date: delivery anomaly executed date
        :param index: index_name in order to get dimension_of data. If there is no dimension, no need to be assigned
        :return: data frame
        """
        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_name": 'delivery_anomaly'}},
                           {"term": {"index": get_index_group(self.order_index)}},
                           {"term": {"report_types.type": anomaly_type}},
                           {'range': {'report_date': {
                               'lt': current_date_to_day().isoformat() if start_date is None else start_date}}}]

        self.query_es = QueryES(port=self.port,
                                host=self.host)
        self.query_es.query_builder(fields=None, _source=True,
                                    date_queries=date_queries,
                                    boolean_queries=boolean_queries)
        _res = self.query_es.get_data_from_es(index="reports")
        return pd.DataFrame(_res[0]['_source']['data'])









