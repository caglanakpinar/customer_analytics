import numpy as np
import pandas as pd
import sys, os, inspect
import datetime

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_es_port, default_es_host, default_query_date
from customeranalytics.utils import convert_to_date, current_date_to_day, convert_to_iso_format, get_index_group
from customeranalytics.utils import calculate_time_diff, dimension_decision
from customeranalytics.data_storage_configurations.query_es import QueryES


class RFM:
    """
    RFM is a generally useful technique in order to classify customers according to their engagement with the business
    Recency      : Last time attraction to the business.
    Frequency(F) : how frequently users are engaged in the business.
    Monetary(M)  : Average amount(value) per customer.
    !!! The users, who have only 1 order will not be included in calculations !!!

        !!!!
        ******* ******** *****
        Dimensional RFM:
        RFM values must be created individually for dimensions. For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimensions can be assigned in order to create the RFM values.

        download_index; downloads_location1 this will be the location dimension of
                        parameters in order to query downloads indexes; 'location1'.
        download_index; orders_location1 this will be the location dimension of
                        parameters in order to query orders indexes; 'location1'.
        ******* ******** *****
        !!!
    """

    def __init__(self,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """

        !!!!
        ******* ******** *****
        Dimensional RFM:
        RFM values must be created individually for dimensions. For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimensions can be assigned in order to create the RFM values.

        download_index; downloads_location1 this will be the location dimension of
                        parameters in order to query downloads indexes; 'location1'.
        download_index; orders_location1 this will be the location dimension of
                        parameters in order to query orders indexes; 'location1'.
        ******* ******** *****
        !!!

        :param download_index: elasticsearch port
        :param order_index: elasticsearch port
        :param host: elasticsearch host
        :param port: elasticsearch port
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.download_index = download_index
        self.order_index = order_index
        self.query_es = QueryES(port=port, host=host)
        self.orders_field_data = ["id", "session_start_date", "client", "payment_amount"]
        self.orders = pd.DataFrame()
        self.client_frequency = pd.DataFrame()
        self.client_recency = pd.DataFrame()
        self.client_monetary = pd.DataFrame()
        self.rfm = pd.DataFrame()
        self.max_order_date = datetime.datetime.now()

    def dimensional_query(self, boolean_query=None):
        if dimension_decision(self.order_index):
            if boolean_query is None:
                boolean_query = [{"term": {"dimension": self.order_index}}]
            else:
                boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def get_data(self, start_date=None):
        """
        query orders index to collect the data with columns which are "session_start_date", "client", "payment_amount".
        :param start_date:
        :return:
        """
        start_date = default_query_date if start_date is None else start_date
        if len(self.orders) == 0:
            self.query_es = QueryES(port=self.port, host=self.host)
            self.query_es.query_builder(fields=self.orders_field_data,
                                        boolean_queries=self.dimensional_query([{"term": {"actions.purchased": True}}]),
                                        date_queries=[{"range": {"session_start_date": {"gte": start_date}}}])
            self.orders = pd.DataFrame(self.query_es.get_data_from_es())
            self.orders['date'] = self.orders['session_start_date'].apply(lambda x: convert_to_date(x))

    def frequency(self):
        """
        Frequency of users;
            -   assign dates of next orders per user as a column.
                So, each row will have a current order date and the next order date per user.
            -   Calculate the hour difference from the current order date to the next order date.
            -   Calculate the average hourly difference per user.
        User has only 1 order will not be included in calculations.
        """
        self.orders['next_order_date'] = self.orders.sort_values(
            by=['client', 'date'], ascending=True).groupby(['client'])['date'].shift(-1)
        self.orders['diff_hours'] = self.orders.apply(
            lambda row: calculate_time_diff(row['date'], row['next_order_date'], 'hour'), axis=1)
        self.client_frequency = self.orders.query("next_order_date == next_order_date").groupby("client").agg(
            {"diff_hours": "mean"}).reset_index().rename(columns={"diff_hours": "frequency"})

    def recency(self):
        """
        The recency of users;
            -   Calculate the last transaction (purchased) date of the whole population.
            -   Find each user of maximum transaction (purchased) date.
            -   Calculate the hour difference from the maximum transaction date to each user of the maximum transaction date.
        """
        self.max_order_date = max(self.orders['date'])
        self.client_recency = self.orders.groupby("client").agg({"date": "max"}).reset_index()
        self.client_recency['recency'] = self.client_recency.apply(
            lambda row: calculate_time_diff(row['date'], self.max_order_date, 'hour'), axis=1)
        self.client_recency = self.client_recency.drop('date', axis=1)

    def monetary(self):
        """
        Monetary of users;
            -   Calculate the average purchased amount per user
        """
        self.client_monetary = self.orders.groupby("client").agg({"payment_amount": "mean"}).reset_index().rename(
            columns={"payment_amount": "monetary"})

    def execute_rfm(self, start_date):
        """
        1.  Execute R, F, M calculations.
        2.  Merge data-frames (R, F, M data-frames).
        3.  Insert into the reports index with report_name 'rfm'.
        """
        self.get_data(start_date=start_date)
        self.frequency()
        self.recency()
        self.monetary()
        self.rfm = pd.merge(self.client_frequency, self.client_recency, on='client', how='left')
        self.rfm = pd.merge(self.rfm, self.client_monetary, on='client', how='left')
        self.insert_into_reports_index(self.rfm, start_date, index=self.order_index)

    def insert_into_reports_index(self, rfm, start_date, index='orders'):
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
                        "report_name": "rfm",
                        "index": get_index_group(index),
                        "report_types": {},
                        "data": rfm.fillna(0).to_dict("results")}]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def fetch(self, start_date=None):
        """
        Collect RFM values for each user. Collecting stored RFM is useful in order to initialize Customer Segmentation.
        :return: data-frame
        """

        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_name": "rfm"}}, {"term": {"index": get_index_group(self.order_index)}}]

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




