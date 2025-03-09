import numpy as np
import pandas as pd
import datetime

from customeranalytics.exploratory_analysis.base import BaseEDA


class RFM(BaseEDA):
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

    def __init__(self, host=None, port=None, download_index='downloads', order_index='orders'):
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
        super().__init__(host, port, download_index, order_index)
        self.download_index = download_index
        self.order_index = order_index
        self.orders_field_data = ["id", "session_start_date", "client", "payment_amount"]
        self.orders = pd.DataFrame()
        self.client_frequency = pd.DataFrame()
        self.client_recency = pd.DataFrame()
        self.client_monetary = pd.DataFrame()
        self.rfm = pd.DataFrame()
        self.max_order_date = datetime.datetime.now()


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
            lambda row: self.calculate_time_diff(row['date'], row['next_order_date'], 'hour'), axis=1)
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
            lambda row: self.calculate_time_diff(row['date'], self.max_order_date, 'hour'), axis=1)
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
        self.insert_into_reports_index(
            "rfm",
            eda=self.rfm,
            start_date=start_date,
            index=self.order_index,
        )






