import numpy as np
import pandas as pd
import sys, os, inspect
import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.grid.grid_search import H2OGridSearch


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_es_port, default_es_host
from customeranalytics.utils import *
from customeranalytics.data_storage_configurations.query_es import QueryES
from customeranalytics.exploratory_analysis import query_exploratory_analysis, ea_configs


class CustomerSegmentation:
    """
    Customer Segmentation is one of the crucial problems for Businesses that are mostly engaging with their buyers.
    This relationship for some reason might differ from one customer to another one.
    By using RFM (Recency - Frequency Monetary) each customer can be segmented by using Clustering Algorithms.

    - Data Manipulation:
        - It has been already applied on rfm.py. Collect data from reports index.
        - Segment Each Metric individually by using KMeans with k = 5.
          Why is k decided as 5?
          - It is related to the Segmentation rule after assigning the segmentation process.
          - There, Each cluster number named to a human-readable name rather than 1, 2, ..5.

        - In order to optimize parameters of KMeans (except k). GRidSearch is applied.
        - After Clustering Process has been done for R, F, and M, each cluster of average value sorted and
          cluster labels (1, 2, 3, 4, 5) are reassigned in order. label 5 will be the highest, 1 will be the lowest.
        - At the end, the user is the label in human-readable format;
            "champions": {'r': [5], 'f': [5], 'm': [5]},
            "loyal_customers": {'r': [3, 4, 5], 'f': [3, 4, 5], 'm': [5, 4, 3, 3]},
            "potential loyalist": {'r': [5, 4, 3], 'f': [5, 4, 3], 'm': [4, 3, 2]},
            "new customers": {'r': [5], 'f': [2, 1], 'm': [1, 2, 3, 4, 5]},
            "promising": {'r': [3, 4], 'f': [1, 2], 'm': [1, 2]},
            "need attention": {'r': [3, 2], 'f': [3, 2], 'm': [3, 2]},
            "about to sleep": {'r': [3, 4], 'f': [1, 2], 'm': [1, 2]},
            "at risk": {'r': [2], 'f': [5, 4, 3], 'm': [5, 4, 3]},
            "can`t lose them": {'r': [1, 2], 'f': [5], 'm': [5]},
            "hibernating": {'r': [2, 3], 'f': [2], 'm': [2]},
            "lost": {'r': [1], 'f': [1, 2, 3, 4, 5], 'm': [1, 2, 3, 4, 5]}}

    """
    def __init__(self,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        ******* ******** *****
        Dimensional Customer Segmentation:
        Descriptive Statistics must be created individually for dimensions.
        For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimension can be assigned in order to create the Segmentations

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
        self.rfm = pd.DataFrame()
        self.METRICS = ['recency', 'frequency', 'monetary']
        self.METRIC_VALUES = {'recency': ['recency', 'rec', 'r'],
                              'frequency': ['frequency', 'freq', 'f'],
                              'monetary': ['monetary', 'mon', 'm']
                              }
        self.Z_VALUES = [-1.96, 1.96]  # 95% OF CONFIDENCE
        self.clustering_parameters = {'split_ratio': 0.8, 'seed': 1234, 'k': 5,
                                      'hyper_params': {'standardize': [True, False],
                                                       'init': ['Random', 'Furthest', 'PlusPlus']},
                                      'search_criteria': {'strategy': "Cartesian"}
                                      }
        self.accepted_minimum_prob = 0.05
        self.frequency_segments = {}
        self.recency_segments = {}
        self.monetary_segments = {}
        self.customer_segments = {"champions": {'r': [5], 'f': [5], 'm': [5]},
                                  "loyal customers": {'r': [3, 4, 5], 'f': [3, 4, 5], 'm': [5, 4, 3, 3]},
                                  "potential loyalist": {'r': [5, 4, 3], 'f': [5, 4, 3], 'm': [4, 3, 2]},
                                  "new customers": {'r': [5], 'f': [2, 1], 'm': [1, 2, 3, 4, 5]},
                                  "promising": {'r': [3, 4], 'f': [1, 2], 'm': [1, 2]},
                                  "need attention": {'r': [3, 2], 'f': [3, 2], 'm': [3, 2]},
                                  "about to sleep": {'r': [3, 4], 'f': [1, 2], 'm': [1, 2]},
                                  "at risk": {'r': [2], 'f': [5, 4, 3], 'm': [5, 4, 3]},
                                  "can`t lose them": {'r': [1, 2], 'f': [5], 'm': [5]},
                                  "hibernating": {'r': [2, 3], 'f': [2], 'm': [2]},
                                  "lost": {'r': [1], 'f': [1, 2, 3, 4, 5], 'm': [1, 2, 3, 4, 5]}}
        self.segments_numerics = {}
        self.detected_segments = []
        self.insert_columns = ["client", "recency_segment", "frequency_segment",
                               "monetary_segment", "segments", "segments_numeric"]

    def get_rfm(self, date):
        """
        RFM values for segmentation can be fetched from the reports index with related dimensions.
        """
        ea_configs['rfm']['order_index'] = self.order_index
        _total_cols, _counter = 0, 0
        while len(set(self.rfm.columns) & set(self.METRICS)) != 3:
            self.rfm = query_exploratory_analysis(ea_configs, {"start_date": date}, "rfm")
            if _counter >= 10:
                _total_cols = 3
            else:
                _total_cols = len(set(self.rfm.columns) & set(self.METRICS))
            _counter += 1

    def segmentation(self, data, metric):
        """
        Each RFM segmentation will be k = 1,.. , 5
        Step of Clustering Process;
            1.  Convert rfm data-frame into the H2O data-frame
            2.  Split data for train and validation.
            3.  Create KMeans process.
            4.  Create GridSearch for KMeans parameters. These are;
                a.  standardize;  standardize the numeric columns to have a mean of zero and unit variance. Standardization is highly recommended;
                b.  split ratio; train validation split ratio, the lower ratio can decrease Underfitting Problem
                c.  init; 'Random', 'Furthest', 'PlusPlus'
                d.  search criteria - strategy;
        These segments are later on CustomerSegmentation will be sorted according to theirs.
        :param data: rfm data-frame with columns recency or monetary or frequency
        :param metric: recency or monetary or frequency
        :return:
        """
        h2o.init()  # initialize the H2O environment
        rfm_data = h2o.H2OFrame(data)  # convert pandas data-frame to H2O data-frame
        # split train and test data
        train, valid = rfm_data.split_frame(ratios=[self.clustering_parameters['split_ratio']],
                                            seed=self.clustering_parameters['seed'])
        # build KMeans algorithm
        rfm_kmeans = H2OKMeansEstimator(k=self.clustering_parameters['k'],
                                        seed=self.clustering_parameters['seed'],
                                        max_iterations=int(len(data) / 2))
        rfm_kmeans.train(x=metric, training_frame=train, validation_frame=valid)
        # grid search for KMeans parameters
        grid = H2OGridSearch(model=rfm_kmeans, hyper_params=self.clustering_parameters['hyper_params'],
                             search_criteria=self.clustering_parameters['search_criteria'])
        # train using the grid
        grid.train(x=metric, training_frame=train, validation_frame=valid)
        # sort the grid models by total within cluster sum-of-square error.
        sorted_grid = grid.get_grid(sort_by='tot_withinss', decreasing=False)
        prediction = sorted_grid[0].predict(rfm_data)
        data = rfm_data.concat(prediction, axis=1)[[metric, 'predict']].as_data_frame(use_pandas=True)
        data = data.rename(columns={'predict': metric + '_segment'})
        # assign segments for each metric (Recency - Monetary  - Frequency)
        data[metric + '_segment'] = data[metric + '_segment'].apply(lambda x: x + 1)
        return data

    def current_day_r_f_m_clustering(self):
        """
        combine segmented values to the recent rfm data frame. After concatenation;

                  client	frequency	recency	monetary	recency_segment	frequency_segment	monetary_segment
        6115	u_174194	0.0	        1663	32.8500	    2	            2	                3
        10060	u_220840	1.0	        342	    22.4700	    4	            2	                4
        """
        for metric in self.METRICS:
            self.rfm = pd.concat([self.rfm, self.segmentation(self.rfm[[metric]], metric).drop(metric, axis=1)], axis=1)

    def get_frequency_segments(self):
        """
        Each segmented label is assigned as unsorted format.
        Here finding the cluster centroid for frequency clusters (it is calculated via average at this process).
        e.g. centroids - labels pairs  7.2 - 5, 4.2 - 2, 30.2 - 4, 1.3 - 3, 44 - 1,
             sorted_labels;
             previous_labels  sorted_labels centroids
             3                1             1.3
             2                2             4.2
             5                3             7.2
             4                4             30.2
             1                5             44
        """
        for i in self.rfm.groupby("frequency_segment").agg(
                                {"frequency": "mean"}).reset_index().sort_values(
                                by='frequency', ascending=True).reset_index(drop=True).reset_index().to_dict('results'):
            self.frequency_segments[i['frequency_segment']] = 5 - i['index']

    def get_recency_segments(self):
        """
        Each segmented label is assigned as unsorted format.
        Here finding the cluster centroid for recency clusters (it is calculated via average at this process).
        e.g. centroids - labels pairs  7.2 - 5, 4.2 - 2, 30.2 - 4, 1.3 - 3, 44 - 1,
             sorted_labels;
             previous_labels  sorted_labels centroids
             3                1             1.3
             2                2             4.2
             5                3             7.2
             4                4             30.2
             1                5             44

        """
        for i in self.rfm.groupby("recency_segment").agg(
                                {"recency": "mean"}).reset_index().sort_values(
                                by='recency', ascending=True).reset_index(drop=True).reset_index().to_dict('results'):
            self.recency_segments[i['recency_segment']] = 5 - i['index']

    def get_segments_numeric(self):
        """
        format; {"champions": 0, ..... , "can`t lose them": 8}
        """
        self.segments_numerics = {s[0]: s[1] for s in zip(list(self.customer_segments.keys()) + ['others'],
                                                          list(range(len(list(self.customer_segments.keys()))+1)))}

    def get_monetary_segments(self):
        """
        Each segmented label is assigned as unsorted format.
        Here finding the cluster centroid for monetary clusters (it is calculated via average at this process).
        e.g. centroids - labels pairs  7.2 - 5, 4.2 - 2, 30.2 - 4, 1.3 - 3, 44 - 1,
             sorted_labels;
             previous_labels  sorted_labels centroids
             3                1             1.3
             2                2             4.2
             5                3             7.2
             4                4             30.2
             1                5             44

        """
        for i in self.rfm.groupby("monetary_segment").agg(
                                {"monetary": "mean"}).reset_index().sort_values(
                                by='monetary', ascending=True).reset_index(drop=True).reset_index().to_dict('results'):
            self.monetary_segments[i['monetary_segment']] = 5 - i['index']

    def combine_segments(self):
        """
        apply sorted segments
        """
        self.rfm['frequency_segment'] = self.rfm['frequency_segment'].apply(lambda x: self.frequency_segments[x])
        self.rfm['recency_segment'] = self.rfm['recency_segment'].apply(lambda x: self.recency_segments[x])
        self.rfm['monetary_segment'] = self.rfm['monetary_segment'].apply(lambda x: self.monetary_segments[x])

    def detect_customer_segments(self):
        """
        Human Readable Form Of Segments;
        After the segmentation process, There will be 5 * 5 * 5 = 125 individual segments will be created.
        e.g. 1 - 3 - 5 (r - f - m)
        Now, another topic is how segmentation outputs can be converted to clearly understandable segments.
        By searching on articles about RFM segmentation, one the usual segments are;
            - champions
            - loyal_customers
            - potential loyalist
            - new customers
            - promising
            - need attention
            - about to sleep
            - at risk
            - can`t lose them
            - hibernating
            - lost
        Each segment of label ranges are assigned to the variable 'self.customer_segments'.
        """
        total_clients = list(self.rfm['client'].unique())
        for s in self.customer_segments:
            _segment_clients = list(self.rfm[(self.rfm['recency_segment'].isin(self.customer_segments[s]['r'])) &
                                             (self.rfm['frequency_segment'].isin(self.customer_segments[s]['f'])) &
                                             (self.rfm['monetary_segment'].isin(self.customer_segments[s]['m'])) & (
                                              self.rfm['client'].isin(total_clients))]['client'])
            self.detected_segments += list(zip(_segment_clients, (len(_segment_clients) * [s])))
            total_clients = list(set(total_clients) - set(_segment_clients))

    def insert_into_reports_index(self, segments, date=None, index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": date or current date,
         "report_name": "segmentation",
         "index": "main",
         "report_types": {},
         "data": segments.fillna(0.0).to_dict("results") -  dataframe to list of dictionary
         }
         !!! null values are assigned to 0.

        :param segments: data set, data frame
        :param index: dimensionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = [{"id": np.random.randint(200000000),
                        "report_date": current_date_to_day().isoformat() if date is None else date,
                        "report_name": "segmentation",
                        "index": get_index_group(index),
                        "report_types": {},
                        "data": segments.fillna(0.0).to_dict("results")}]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def execute_customer_segment(self, start_date=None):
        """
        Process of the Customer Segmentation
            1.  fetch rfm calculated values from reports index.
            2.  create KMeans Segmentation.
            3.  sort labels related to centroids for R, F, and M individually.
            4.  combine sorted labels.
            6.  assign human-readable segments instead of 1, 2, ..5 labels.
            7.  insert human-readable segments and number labels of each segment into the reports index.
                reports index must be from right dimensional, related orders_index.

        :param start_date: date of reporting date
        :return:
        """
        self.get_rfm(date=start_date)
        self.current_day_r_f_m_clustering()
        self.get_frequency_segments()
        self.get_recency_segments()
        self.get_monetary_segments()
        self.combine_segments()
        self.detect_customer_segments()
        self.get_segments_numeric()

        self.rfm = pd.merge(self.rfm,
                            pd.DataFrame(self.detected_segments).rename(columns={0: "client", 1: "segments"}),
                            on='client', how='left')
        self.rfm['segments'] = self.rfm['segments'].fillna('others')
        self.rfm['segments_numeric'] = self.rfm['segments'].apply(lambda x: self.segments_numerics[x])

        self.insert_into_reports_index(self.rfm[self.insert_columns], date=start_date, index=self.order_index)
        h2o.shutdown(prompt=False)

    def fetch(self, start_date=None):
        """
        Collect Customer segmentation results with the given date.
        :param start_date: customer_segmentation first date
        :return:
        """
        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_name": "segmentation"}},
                           {"term": {"index": get_index_group(self.order_index)}}]

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
            _data = pd.DataFrame(_res[-1]['_source']['data'])
        return _data
