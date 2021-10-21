from os.path import join
import pandas as pd
import numpy as np

import datetime
import pygeohash as gh
from os import listdir
from os.path import dirname, abspath
import networkx as nx
import osmnx as ox
import math

import sys, inspect
currentdir = dirname(abspath(inspect.getfile(inspect.currentframe())))
parentdir = dirname(currentdir)
sys.path.insert(0, parentdir)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json

from tensorflow.keras.optimizers import Adam

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.model_selection import train_test_split

from customeranalytics.data_storage_configurations.query_es import QueryES
from customeranalytics.configs import not_required_default_values, default_es_port, default_es_host, \
    delivery_anomaly_model_parameters, delivery_anomaly_model_hyper_parameters, none_types
from customeranalytics.utils import convert_to_date, dimension_decision, \
    current_date_to_day, find_week_of_monday, read_yaml, write_yaml, find_week_of_monday, \
    convert_dt_to_day_str, convert_to_day, execute_parallel_run


ox.config(use_cache=True, log_console=True)


def haversine(lat1, lon1, lat2, lon2):
    # distance between latitudes
    # and longitudes
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0

    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
         math.cos(lat1) * math.cos(lat2));
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c


def decode_geo_hashed(row):
    x = gh.encode(row['latitude'], row['longitude'], precision=7)
    loc = gh.decode(x)
    return pd.Series([loc[0], loc[1]])


class Routes:
    """
    Delivery Analytics;

    These analysis are only able to be run when delivery data source is created.
    It detects abo-normal durations of rides according to customer location, location of order, hour of the order,
    weekday of the order.

    """
    def __init__(self,
                 temporary_export_path,
                 has_delivery_connection=True,
                 has_pickup_id_lat_lon_connection=True,
                 has_pickup_category_connection=True,
                 has_picker_connection=True,
                 date=convert_to_day(current_date_to_day()),
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
        :param has_pickup_id_lat_lon_connection: has data source pickup_id, pickup_lat, pickup_lon
        :param has_pickup_category_connection: has data source pickup_category
        :param has_picker_connection: has data source picker (delivery person)
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.date = convert_to_day(date)
        self.prev_date_params = self.date - datetime.timedelta(days=30)
        self.prev_date_model = self.date - datetime.timedelta(days=10)
        self.download_index = download_index
        self.order_index = order_index
        self.has_delivery_connection = has_delivery_connection
        self.has_pickup_id_lat_lon_connection = has_pickup_id_lat_lon_connection
        self.has_pickup_category_connection = has_pickup_category_connection
        self.has_picker_connection = has_picker_connection
        self.temporary_export_path = temporary_export_path
        self.query_es = QueryES(port=self.port, host=self.host)
        self.data = pd.DataFrame()
        self.distance = None
        self.centroid = []
        self.centroid = []
        self.G = None
        self.renaming_columns = {'id': 'order_id', 'delivery.delivery_date': 'delivery_date',
                                 'delivery.prepare_date': 'prepare_date', 'delivery.return_date': 'return_date',
                                 'delivery.latitude': 'latitude', 'delivery.longitude': 'longitude',
                                 'delivery.pickup_id': 'pickup_id', 'delivery.pickup_lat': 'pickup_lat',
                                 'delivery.pickup_lon': 'pickup_lon', 'delivery.picker': 'picker',
                                 'delivery.pickup_category': 'pickup_category'}
        self.hp = HyperParameters()
        self.parameter_tuning_trials = 10  # number of parameter tuning trials
        self.directory = join(abspath(""), "delivery_duration_prediction")
        self.params_tuned_file = f"tuned_parameters_{str(self.date)[0:10]}.yaml"
        self.model_file = join(self.directory, f"trained_model_{str(self.date)[0:10]}.json")
        self.weight_file = join(self.directory, f"weight_{str(self.date)[0:10]}.h5")
        self.files = {"tuned_parameter_file": f"tuned_parameters_{str(self.date)[0:10]}.yaml",
                      "model_file": join(self.directory, f"trained_model_{str(self.date)[0:10]}.json"),
                      "weight_file": join(self.directory, f"weight_{str(self.date)[0:10]}.h5")}
        self.accepted_order_count = 5
        self.features = ["haversine_distance", "dur_geo_hashed", "dur_pickup", "picker_throughput",
                         "dur_w", "dur_c", "dur_picker", "delivery_duration", "latitude", "longitude",
                         "pickup_lat", "pickup_lon"]
        self.target = "delivery_duration"
        self.x_columns = list(set(self.features) - set(self.target))
        self.features_norm = ['norm_' + str(i) for i in self.features]
        self.rename_geo_cols = lambda x, y: x.rename(columns={i[1]: i[0] for i in zip(['latitude', 'longitude'] ,y)})
        self.model_data = {}
        self.centroid = []
        self.duration_metrics = ['deliver', 'prepare', 'ride', 'returns']
        self.anomaly_metrics = ["customer", 'location', 'weekday_hour']
        self._model = {}
        self.params = delivery_anomaly_model_parameters
        self.hyper_params = delivery_anomaly_model_hyper_parameters
        self.tuned_params = {}
        self.hp = HyperParameters()
        self.parameter_tuning_trials = 1  # number of parameter tuning trials
        self.renaming_columns = {'id': 'order_id', 'delivery.delivery_date': 'delivery_date',
                                 'delivery.prepare_date': 'prepare_date', 'delivery.return_date': 'return_date',
                                 'delivery.latitude': 'latitude', 'delivery.longitude': 'longitude',
                                 'delivery.pickup_id': 'pickup_id', 'delivery.pickup_lat': 'pickup_lat',
                                 'delivery.pickup_lon': 'pickup_lon', 'delivery.picker': 'picker',
                                 'delivery.pickup_category': 'pickup_category', 'date': 'end_date'}
        self.delivery_fields = list(self.renaming_columns.keys()) + ['client', 'session_start_date']
        self.convert_to_float = lambda x: np.asarray(x).astype('float32')
        self.model_data = {"feature_data": pd.DataFrame(),
                           "train_x": [],
                           "test_x": [],
                           "train_y": [],
                           "text_y": [],
                           'features': self.features,
                           'features_norm': self.features_norm,
                           'model': None,
                           'params': self.params,
                           'hyper_params': self.hyper_params}
        self.min_max = {i: {"min": None, "max": None} for i in self.features}

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
            columns=self.renaming_columns)
        self.data = self.data.query("delivery_date == delivery_date")
        self.data['latitude'] = self.data['latitude'].apply(lambda x: float(x))
        self.data['longitude'] = self.data['longitude'].apply(lambda x: float(x))

    def get_centroid(self):
        self.centroid = [np.mean(self.data['latitude']), np.mean(self.data['longitude'])]

    def get_graph_distance(self):
        self.distance = 1000 # int(haversine(min(self.data['latitude']), min(self.data['longitude']),
                            #           max(self.data['latitude']), max(self.data['longitude'])) * 1000) * 2

    def create_graph(self, transport_mode='drive'):
        """Transport mode = ‘walk’, ‘bike’, ‘drive’, ‘drive_service’, ‘all’, ‘all_private’, ‘none’"""
        self.G = ox.graph_from_point(self.centroid, dist=self.distance, network_type=transport_mode)
        self.G = ox.add_edge_speeds(self.G)  # Impute
        self.G = ox.add_edge_travel_times(self.G)  # Travel time

    def calculate_nodes_travel_times(self, route):
        node_start = []
        node_end = []
        X_to = []
        Y_to = []
        X_from = []
        Y_from = []
        length = []
        travel_time = []
        for u, v in zip(route[:-1], route[1:]):
            node_start.append(u)
            node_end.append(v)
            length.append(round(self.G.edges[(u, v, 0)]['length']))
            travel_time.append(round(self.G.edges[(u, v, 0)]['travel_time']))
            X_from.append(self.G.nodes[u]['x'])
            Y_from.append(self.G.nodes[u]['y'])
            X_to.append(self.G.nodes[v]['x'])
            Y_to.append(self.G.nodes[v]['y'])
        df = pd.DataFrame(list(zip(node_start, node_end, X_from, Y_from, X_to, Y_to, length, travel_time)),
                          columns=['node_start', 'node_end', 'X_from', 'Y_from', 'X_to', 'Y_to', 'length',
                                   'travel_time'])
        return df

    def calculate_deliveries(self, row):
        try:
            # find_start & end node
            start_node = ox.get_nearest_node(self.G, (row['pickup_lat'], row['pickup_lon']))
            end_node = ox.get_nearest_node(self.G, (row['latitude'], row['longitude']))
            # calculate the shortest path
            route = nx.shortest_path(self.G, start_node, end_node, weight='travel_time')
            distance = self.calculate_nodes_travel_times(route)
            distance['order_id'] = row['order_id']
            return pd.Series(
                [distance.to_dict('results'), sum(list(distance['travel_time'])), sum(list(distance['length']))])
        except Exception as e:
           return pd.Series([{}, '-', '-'])

    def calculate_deliveries_v2(self, row):
        try:
            # find_start & end node
            start_node = ox.get_nearest_node(self.G, (row['pickup_lat'], row['pickup_lon']))
            end_node = ox.get_nearest_node(self.G, (row['latitude'], row['longitude']))
            # calculate the shortest path
            route = nx.shortest_path(self.G, start_node, end_node, weight='travel_time')
            distance = self.calculate_nodes_travel_times(route)
            distance['order_id'] = row['order_id']
            delivery_parallel[row['order_id']] = [distance.to_dict('results'),
                                                  sum(list(distance['travel_time'])), sum(list(distance['length']))]
        except Exception as e:
            delivery_parallel[row['order_id']] = [{}, '-', '-']

    def parallel_run_for_deliveries(self):
        global delivery_parallel
        delivery_parallel = {}
        execute_parallel_run(self.data.to_dict('results'),
                             self.calculate_deliveries_v2,
                             parallel=int(len(self.data) / 16),
                             prints=True)

        orders = list(delivery_parallel.keys())
        self.data[['route', 'delivery_duration', 'delivery_distance']] = self.data.apply(
            lambda row: pd.Series(delivery_parallel[row['order_id']])
            if row['order_id'] in orders else pd.Series(['-'] * 3), axis=1)

    def feature_normalization(self):
        for i in self.features:
            _values = self.data[~self.data[i].isin(none_types)][i]
            self.min_max[i]["min"], self.min_max[i]["max"] = min(_values), max(_values)
            _range = self.min_max[i]["max"] - self.min_max[i]["min"]
            self.data['norm_' + i] = self.data[i].apply(
                lambda x: (x - self.min_max[i]["min"]) / _range if x not in none_types else '-')
            self.data = self.data[(~self.data['norm_' + i].isin(none_types)) &
                                  (self.data['norm_' + i] == self.data['norm_' + i])]
        print(self.data.columns)

    def data_prepare(self):
        for dt in ['end_date', 'session_start_date']:
            self.data[dt] = self.data[dt].apply(lambda x: convert_to_date(x) if x is not None else None)
        for loc in ['latitude', 'longitude', 'pickup_lat', 'pickup_lon']:
            self.data[loc] = self.data[loc].apply(lambda x: float(x) if x is not None else None)

        self.data[['lat_perc_7', 'lon_perc_7']] = self.data.apply(lambda row: decode_geo_hashed(row), axis=1)
        self.data['mondays'] = self.data['end_date'].apply(lambda x: find_week_of_monday(x))
        self.data['isoweekday'] = self.data['session_start_date'].apply(lambda x: x.isoweekday())
        self.data['is_weekend'] = self.data['isoweekday'].apply(lambda x: 0 if x in list(range(1, 6)) else 1)
        self.data['hour'] = self.data['session_start_date'].apply(lambda x: x.hour)
        self.data['day'] = self.data['session_start_date'].apply(lambda x: convert_dt_to_day_str(x))

        self.data['haversine_distance'] = self.data.apply(
            lambda row: haversine(row['latitude'], row['longitude'], row['pickup_lat'], row['pickup_lon']),
            axis=1)

        self.parallel_run_for_deliveries()
        # self.data[['route', 'delivery_duration', 'delivery_distance']] = self.data.apply(
        #     lambda row: self.calculate_deliveries(row), axis=1)
        _data = self.data.query("delivery_duration != '-'")
        hour_weekday_duration = _data.groupby(["isoweekday", "hour"]).agg(
            {"delivery_duration": "mean"}).reset_index()
        picker_throughput = _data.groupby(["isoweekday", "hour", "day"]).agg(
            {"picker": lambda x: len(np.unique(x)), "order_id": lambda x: len(np.unique(x))}).reset_index()
        picker_throughput['picker_throughput'] = picker_throughput['order_id'] / picker_throughput['picker']
        picker_throughput = picker_throughput.groupby(["isoweekday", "hour"]).agg({"picker_throughput": "mean"}).reset_index()
        pickup_duration = _data.groupby("pickup_id").agg({"delivery_duration": "mean"}).reset_index()
        week_duration = _data.groupby(["mondays"]).agg({"delivery_duration": "mean"}).reset_index()
        geo_hashed_duration = _data.groupby(["lat_perc_7", "lon_perc_7"]).agg(
            {"delivery_duration": "mean"}).reset_index()
        client_duration = _data.groupby("client").agg({"delivery_duration": "mean"}).reset_index()
        courier_duration = _data.groupby("picker").agg({"delivery_duration": "mean"}).reset_index()

        mergings = [{"df": geo_hashed_duration,
                     "columns": ['lat_perc_7', 'lon_perc_7'],
                     "renaming": {"delivery_duration": "dur_geo_hashed"}},

                    {"df": pickup_duration,
                     "columns": ['pickup_id'],
                     "renaming": {"delivery_duration": "dur_pickup"}},

                    {"df": hour_weekday_duration,
                     "columns": ['isoweekday', 'hour'],
                     "renaming": {"delivery_duration": "dur_hr_w"}},

                    {"df": picker_throughput,
                     "columns": ['isoweekday', 'hour'],
                     "renaming": {}},

                    {"df": week_duration,
                     "columns": ['mondays'],
                     "renaming": {"delivery_duration": "dur_w"}},

                    {"df": client_duration,
                     "columns": ['client'],
                     "renaming": {"delivery_duration": "dur_c"}},

                    {"df": courier_duration,
                     "columns": ['picker'],
                     "renaming": {"delivery_duration": "dur_picker"}}
                    ]

        for m in mergings:
            self.data = pd.merge(self.data, m["df"].rename(columns=m["renaming"]), on=m["columns"], how='left')

        self.feature_normalization()
        self.model_data['feature_data'] = self.data[self.features + self.features_norm]

    def train_test_split(self, prediction=False):
        if prediction:
            self.model_data['prediction'] = self.convert_to_float(
                self.model_data['feature_data'][['norm_' + i for i in self.x_columns]].values)
        else:
            self.model_data['train_x'], self.model_data['test_x'], \
            self.model_data['train_y'], self.model_data['test_y'] = train_test_split(
                self.model_data['feature_data'][['norm_' + i for i in self.x_columns]].values,
                self.model_data['feature_data'][['norm_' + self.target]].values, test_size=0.25)

            for i in ['train_x', 'train_y', 'test_x', 'test_y']:
                self.model_data[i] = self.convert_to_float(self.model_data[i])

    def build_parameter_tuning_model(self, hp):
        _input = Input(shape=(self.model_data['train_x'].shape[1],))
        _unit = hp.Choice('h_l_unit', self.hyper_params['h_l_unit'])
        _layer = Dense(hp.Choice('h_l_unit', self.hyper_params['h_l_unit']),
                       activation=hp.Choice('activation', self.hyper_params['activation'])
                       )(_input)

        for i in range(1, hp.Choice('hidden_layer_count', self.hyper_params['hidden_layer_count'])):
            _unit = _unit / 2
            _layer = Dense(_unit,
                           activation=hp.Choice('activation', self.hyper_params['activation'])
                           )(_layer)

        output = Dense(1, activation='sigmoid')(_layer)
        model = Model(inputs=_input, outputs=output)
        model.compile(loss=self.hyper_params['loss'],
                      optimizer=Adam(lr=hp.Choice('lr', self.hyper_params['lr'])))
        return model

    def build_model(self):
        _input = Input(shape=(self.model_data['train_x'].shape[1],))
        _unit = self.params['h_l_unit']
        _layer = Dense(self.params['h_l_unit'],
                       activation=self.params['activation']
                       )(_input)

        for i in range(1, self.params['hidden_layer_count']):
            _unit = _unit / 2
            _layer = Dense(_unit, activation=self.params['activation'])(_layer)

        output = Dense(1, activation='sigmoid')(_layer)
        self.model_data["model"] = Model(inputs=_input, outputs=output)
        self.model_data["model"].compile(loss=self.params['loss'],
                                         optimizer=Adam(lr=self.params['lr']))
        print(self.model_data["model"].summary())

    def execute_model(self):
        self.build_model()
        self.model_data["model"].fit(self.model_data['train_x'], self.model_data['train_y'],
                                     batch_size=self.params['batch_size'],
                                     epochs=self.params['epochs'],
                                     verbose=1,
                                     validation_data=(self.model_data['train_x'], self.model_data['train_y']),
                                     shuffle=True)
        print(self.model_data["model"])
        self.model_from_to_json(is_writing=True)

    def prediction(self):
        self.data["delivery_duration"] == self.model_data['model'].predict(self.model_data['prediction'])

    def model_from_to_json(self, is_writing=False):
        """
        writing & reading Keras model
        Keras model (.json) and optimized weight matrix (.h5)
        """
        if is_writing:
            model_json = self.model_data["model"].to_json()
            with open(self.files["model_file"], "w") as json_file:
                json_file.write(model_json)
            self.model_data["model"].save_weights(self.files["weight_file"])
        else:
            try:
                json_file = open(self.files["model_file"], 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)
                model.load_weights(self.files["weight_file"])
                return model
            except Exception as e:
                print(e)
                return {}

    def check_for_file(self, type):
        """
        :params type: tuned (parameter tuning), model, weight
        """
        type_splited = type.split("_")[0]
        tuned_params_files = []
        accept = True
        for f in listdir(dirname(self.directory)):
            _f_splited = f.split("_")
            if _f_splited[0] == type_splited:  # if the file is tuned param
                _date = _f_splited[2]
                if self.prev_date_params < convert_to_date(_date):  # if file create in 30 days
                    tuned_params_files.append({"file": f, "date": _date})
        if len(tuned_params_files) != 0:
            accept = False # no need to run parameter tuning/model build
            if len(tuned_params_files) > 1:  # if there is more than once parameter tuning
                # pick latest one
                self.files[type] = list(pd.DataFrame(tuned_params_files).sort_values(
                    "date", ascending=False).to_dict('results')[0]['file'])[0]
            else:
                self.files[type] = tuned_params_files[0]['file']
        return accept

    def execute_ml_process(self):
        self.train_test_split(prediction=True)
        if self.check_for_file(type="tuned_parameter_file") or \
            self.check_for_file(type="model") or \
            self.check_for_file(type="weight"):
            self.train_test_split()

        if self.check_for_file(type="tuned_parameter_file"):
            kwargs = {'directory': self.directory}
            tuner = RandomSearch(self.build_parameter_tuning_model,
                                 max_trials=self.parameter_tuning_trials,
                                 hyperparameters=self.hp,
                                 allow_new_entries=True,
                                 objective='loss', **kwargs)
            tuner.search(x=np.asarray(self.model_data['train_x']).astype('float32'),
                         y=np.asarray(self.model_data['train_y']).astype('float32'),
                         epochs=5,
                         batch_size=self.hyper_params['batch_size'],
                         verbose=1,
                         validation_data=(np.asarray(self.model_data['test_x']).astype('float32'),
                                          np.asarray(self.model_data['test_y']).astype('float32')))
            for p in tuner.get_best_hyperparameters()[0].values:
                if p in list(self.params.keys()):
                    self.params[p] = tuner.get_best_hyperparameters()[0].values[p]

            write_yaml(self.directory, self.files["tuned_parameter_file"], self.params, ignoring_aliases=False)
        else:
            self.params = read_yaml(self.directory, self.files["tuned_parameter_file"])

        if self.check_for_file(type="model") or self.check_for_file(type="weight"):
            self.execute_model()
        else:
            self.prediction()

    def execute_models(self):
        if self.has_pickup_id_lat_lon_connection and self.has_delivery_connection:
            self.get_delivery_data()
            self.get_centroid()
            self.get_graph_distance()
            self.create_graph()
            self.data_prepare()
            self.execute_ml_process()

