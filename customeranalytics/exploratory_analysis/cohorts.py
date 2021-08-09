import numpy as np
import pandas as pd
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_es_port, default_es_host, default_query_date
from customeranalytics.utils import *
from customeranalytics.data_storage_configurations.query_es import QueryES


class Cohorts:
    """
        - From Download to 1st Order Cohort.
            - There are two types of that cohort. Weekly and Daily.

        ***** Examples of Interpret Cohorts; ******

        Download to 1st Order Daily Cohort;

        	daily	        0	    1	    2	   3	   4	    118	119	120	121
            2021-01-18	501.0	351.0	529.0	393.0	404.0	...	0.0	0.0	0.0	0.0
            2021-01-19	535.0	356.0	356.0	504.0	412.0	...	0.0	0.0	0.0	0.0
            2021-01-20	608.0	375.0	387.0	389.0	564.0	...	0.0	0.0	0.0	0.0
            2021-01-21	422.0	278.0	312.0	320.0	295.0	...	0.0	0.0	0.0	0.0

        - 0 - 2021-01-18 ; 501.0 Clients who download in 2021-01-18, have first order in same day
        - 2 - 2021-01-21 ; 312.0 Clients who download in 2021-01-18, have first order after 2 days.

        Orders From 1st to 2nd Weekly Cohort;

            weekly	         0	     1	    2	     3	       9	 10	  11
            2021-01-18	1586.0	1232.0	877.0	1050.0	 ... 0.0	0.0	 0.0
            2021-01-25	1054.0	874.0	1121.0	1070.0	 ... 0.0	0.0	 0.0
            2021-02-01	631.0	876.0	838.0	810.0	 ... 0.0	0.0	 0.0

        - 0 - 2021-01-18 ; 1586.0 Clients who have their 1st orders within the week 2021-01-18 (Monday of each week),
                           have their 2nd order in the same week (within the week 2021-01-18)
        - 2 - 2021-02-01 ; 838.0 Clients who have their 1st orders within the week 2021-02-01 (Monday of each week),
                           have their 2nd order in after 2 weeks (within the week 2021-02-15)

    Customers Journey Calculation;
    1.  Calculate average Hour difference from Download to 1st orders.
    2.  Calculate average order
    3.  For each calculated average orders, calculate the average purchase amount,
        Example;
        average 2 orders, 1st orders avg 30.3£, 2nd orders avg 33.3£
    4.  Calculate average recent hours customers last order to a recent date.

    """
    def __init__(self,
                 has_download=True,
                 host=None,
                 port=None,
                 download_index='downloads',
                 order_index='orders'):
        """
        !!!!
        ******* ******** *****
        Dimensional Cohorts:
        Cohorts must be created individually for dimensions. For instance, the Data set contains locations dimension.
        In this case, each location of 'orders' and 'downloads' indexes must be created individually.
        by using 'download_index' and 'order_index' dimensions can be assigned in order to create a cohort.

        download_index; downloads_location1 this will be the location dimension of
                        parameters in order to query downloads indexes; 'location1'.
        download_index; orders_location1 this will be the location dimension of
                        parameters in order to query orders indexes; 'location1'.
        ******* ******** *****
        !!!

        !!!!
        If a business has no download action (no mobile app) there is no need for download to first-order calculations.
        !!!!

        :param has_download:  True/False related business of requirements
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
        self.has_download = has_download
        self.download_field_data = ["id", "download_date", "client"]
        self.session_orders_field_data = ["id", "session_start_date", "client", "payment_amount"]
        self.downloads = pd.DataFrame()
        self.orders = pd.DataFrame()
        self.download_to_first_order = pd.DataFrame()
        self.time_periods = ['daily', 'weekly']
        self.cohorts = {'downloads_to_1st_order': {_t: None for _t in self.time_periods},
                        'orders_from_1_to_2': {_t: None for _t in self.time_periods},
                        'orders_from_2_to_3': {_t: None for _t in self.time_periods},
                        'orders_from_3_to_4': {_t: None for _t in self.time_periods},
                        'customers_journey': {'hourly': None}
                        }

        self.order_seq = [1, 2, 3]

    def get_time_period(self, transactions, date_column):
        """
        converting date column of  values into the time_periods (hourly weekly, monthly,..)
        :param transactions: total data (orders/downloads data with actions)
        :return: data set with time periods
        """
        for p in list(zip(self.time_periods,
                          [convert_dt_to_day_str, find_week_of_monday])):
            transactions[p[0]] = transactions[date_column].apply(lambda x: p[1](x))
        return transactions

    def dimensional_query(self, boolean_query):
        if dimension_decision(self.order_index):
            boolean_query += [{"term": {"dimension": self.order_index}}]
        return boolean_query

    def get_data(self, start_date):
        """
        Collecting orders and downloads data.
        Before query for orders and downloads it checks is there any queried data before.
        query_es.py handles collecting data.
        :param start_date: starting query date
        """
        start_date = default_query_date if start_date is None else start_date
        if len(self.orders) == 0:
            self.query_es = QueryES(port=self.port, host=self.host)
            self.query_es.query_builder(fields=self.session_orders_field_data,
                                        boolean_queries=self.dimensional_query([{"term": {"actions.purchased": True}}]),
                                        date_queries=[{"range": {"session_start_date": {"gte": start_date}}}])
            self.orders = self.query_es.get_data_from_es()
            self.orders = self.get_time_period(pd.DataFrame(self.orders), 'session_start_date')
        if len(self.downloads) == 0:
            if self.has_download:

                self.query_es = QueryES(port=self.port, host=self.host)
                self.query_es.query_builder(fields=self.download_field_data)
                self.downloads = pd.DataFrame(self.query_es.get_data_from_es(index='downloads'))
                # for the dimensional it is only calculating for dimension of users.
                if dimension_decision(self.order_index):
                    self.downloads = self.downloads[self.downloads['client'].isin(list(self.orders['client'].unique()))]
                self.downloads = self.get_time_period(self.downloads, 'download_date')
                self.downloads['download_date'] = self.downloads['download_date'].apply(lambda x: convert_to_date(x))

    def convert_cohort_to_readable_form(self, cohort, time_period, time_period_back=None):
        """
        handles the multi-index problem of the pandas data frame. Each cohort has a date column.
        1. splits date column and creates separate data frame.
        2. collects value columns (0, 1, ..110) same ordered as date data-frame and creates separate date-frame
        3. Concatenate date data-frame and values data-frame.

        time_period_back;
        When only the last 2 weeks of cohort number of assigned on time_period_back will filter out the cohort.


        :param cohort: cohort with multi-index
        :param time_period: weeky, daily, indicated date column
        :param time_period_back: desire time period (# of days)
        :return: data frame with columns; weekly/daily, 0, 1, 2, ... 100
        """
        _time_periods = pd.DataFrame(list(cohort[cohort.columns[0]])).rename(columns={0: time_period})
        _cohort = pd.DataFrame(np.array(cohort.drop(cohort.columns[0], axis=1)))

        if time_period_back is not None:
            time_period_columns, time_period_row = time_period_back, time_period_back
            if max(list(_cohort.columns)) < time_period_back:
                time_period_columns = max(list(_cohort.columns))

            if len(_time_periods) < time_period_back:
                time_period_row = max(list(_cohort.columns))

            _cohort = _cohort[range(time_period_columns)]
            cohort = pd.concat([_time_periods, _cohort], axis=1).tail(time_period_row)
        else:
            cohort = pd.concat([_time_periods, _cohort], axis=1)
        return cohort

    def cohort_download_to_1st_order(self):
        """
        From Download to 1st order weekly and daily event values.
        1. Each user of the first-order date is created as a data-frame.
        2. Each client of days/weeks between download date and 1st order date is calculated individually.
        3. Date column is created (daily or weekly)
        4. pivoted data;
            - columns; weekday difference between download to 1 order date per user.
            - rows; time period; daily or weekly;
            - values; the number of client count

        !!!!
        If a business has no download action (no mobile app) there is no need for download to first-order calculations.
        !!!!

        """
        try:
            if self.has_download:
                self.download_to_first_order = pd.merge(self.orders,
                                                        self.downloads,
                                                        on='client',
                                                        how='left')
                self.download_to_first_order = self.get_time_period(self.download_to_first_order, 'session_start_date')
                for p in self.time_periods:
                    self.download_to_first_order['downloads_to_first_order_' + p] = self.download_to_first_order.apply(
                        lambda row: calculate_time_diff(row['download_date'], row[p], period=p), axis=1)
                    self.cohorts['downloads_to_1st_order'][p] = self.download_to_first_order.sort_values(
                    by=['downloads_to_first_order_' + p, p],
                    ascending=True).pivot_table(index=p,
                                                columns='downloads_to_first_order_'+p,
                                                aggfunc={"client": lambda x: len(np.unique(x))}
                                                ).reset_index().rename(columns={"client": "client_count"})
                    self.cohorts['downloads_to_1st_order'][p] = self.convert_cohort_to_readable_form(
                    self.cohorts['downloads_to_1st_order'][p], time_period=p)
        except Exception as e:
            print(e)

    def get_order_cohort(self, order_seq_num, time_period='daily'):
        """
        1.  Remove users who only have 1 order.
            The main aim here is to create a cohort related to users of 2nd, 3rd and 4th orders.
        2.  Filter the .. the order related to 'order_seq_num'.
        3.  Create cohort; rows are dates (weekly/daily), columns number of date difference (week or day)


        Example of creation a cohort;
            Let`s check the cohort_orders_from_1_to_2_daily;

            order_date  next_order_date    date_diff   client
            2021-01-18  2021-01-21         3           c_20

            This data-frame must be pivoted related to columns order_date,  date_diff, the client (count aggregation).

        :param order_seq_num: 2, 3, 4
        :param time_period: weekly or daily
        :return: cohort data-frame with multi-index
        """
        index_column = time_period if time_period == 'daily' else 'weekly'
        column_pv = 'diff_days' if time_period == 'daily' else 'diff_weeks'
        orders_from_to = self.orders.query("next_order_date == next_order_date")  # removing clients only have 1 order
        orders_from_to = orders_from_to.query("order_seq_num in @order_seq_num")  # filter which cohort is executed
        orders_from_to = orders_from_to.sort_values(by=[column_pv, index_column], ascending=True)
        orders_from_to = orders_from_to.pivot_table(index=index_column,  # rows are dates (days or mondays of each week)
                                                    columns=column_pv,  # day or week difference (integer start with 0)
                                                    aggfunc={"client": lambda x: len(np.unique(x))}  # of unique clients
                                                    ).reset_index().rename(columns={"client": "client_count"})
        return orders_from_to

    def cohort_time_difference_and_order_sequence(self, time_period):
        """
        It is for the ƒinal shape of the cohort related to Orders.
            1.  order sequence number is created as a column per user.
            2.  next order date is assigned as a newly generated column.
            3.  Calculating the time differences between recent order dates and next order dates per user.
                Time difference must be measured related to time periods.
                e.g. daily; day difference, weekly; week difference

        :param time_period: daily, weekly
        """
        if 'order_seq_num' not in list(self.orders.columns):
            self.orders['order_seq_num'] = self.orders.sort_values(
                by=['client', time_period], ascending=True).groupby(['client'])['client'].cumcount() + 1
        if 'next_order_date' not in list(self.orders.columns):
            self.orders['next_order_date'] = self.orders.sort_values(
                by=['client', time_period], ascending=True).groupby(['client'])[time_period].shift(-1)
        if 'diff_days' not in list(self.orders.columns):
            self.orders['diff_days'] = self.orders.apply(
                lambda row: calculate_time_diff(row[time_period], row['next_order_date'], 'daily'), axis=1)
        if 'diff_weeks' not in list(self.orders.columns):
            self.orders['diff_weeks'] = self.orders.apply(
                lambda row: calculate_time_diff(row[time_period], row['next_order_date'], 'weekly'), axis=1)

    def cohort_from_to_order(self):
        """
        From Order .. to order ... weekly and daily event values.
            - Orders From 1st to 2nd Weekly/Daily
            - Orders From 2nd to 3rd Weekly/Daily
            - Orders From 3rd to 4th Weekly/Daily
        !!! check 'get_order_cohort'
        1. Each client of days/weeks between recent order date and previous order date is calculated individually.
            Example; Orders From 1st to 2nd Weekly/Daily is being calculated;
                     Clients orders from 1st to 2nd date differences are calculated
                     according to time period (weekly/daily).
            !!! check 'cohort_time_difference_and_order_sequence'
        2. Date column is created (daily or weekly)
           Related to example above, date column will 1st order dates of clients
        3. pivoted data;
            - columns; week - day difference between from ...th order to .. th order date per user.
            - rows; time period; daily or weekly;
            - values; number of client count
        4. This process is initialized iteratively for (1st, 2nd, 3rd, 4th order).

        """
        for o in self.order_seq:
            for p in self.time_periods:
                try:
                    print("order_seq :", o, " || time_periods :", p)
                    _cohort_name = "orders_from_" + str(o) + "_to_" + str(o+1)
                    self.cohort_time_difference_and_order_sequence(p)
                    self.cohorts[_cohort_name][p] = self.get_order_cohort(order_seq_num=[o], time_period=p)
                    self.cohorts[_cohort_name][p] = self.convert_cohort_to_readable_form(self.cohorts[_cohort_name][p],
                                                                                         time_period=p)
                except Exception as e:
                    print(e)

    def customer_average_journey(self):
        """
        Customers Journey Calculation;
        1.  Calculate average Hour difference from Download to 1st orders.
        2.  Calculate average order
        3.  For each calculated average orders, calculate the average purchase amount,
            Example;
            average 2 orders, 1st orders avg 30.3£, 2nd orders avg 33.3£
        4.  Calculate average recent hours customers last order to a recent date.
            """
        try:
            # convert session start date to datetime format
            self.orders['session_start_date'] = self.orders['session_start_date'].apply(lambda x: convert_to_date(x))
            # e.g. the average is 3; customers of 1st, 2nd 3rd orders have involved the process.
            avg_order_count = int(np.mean(self.orders['order_seq_num']))
            # max date for calculate average recency value (hour).
            max_date = max(self.orders['session_start_date'])
            # first orders per customer
            self.orders['first_order_date'] = self.orders['session_start_date']
            self.orders['last_order_date'] = self.orders['session_start_date']
            first_last_orders = self.orders.groupby("client").agg(
                {"first_order_date": "min", "last_order_date": "max", "order_seq_num": "max"}).reset_index().rename(
                columns={"order_seq_num": "max_order_seq"})
            first_last_orders = pd.merge(self.downloads.drop('id', axis=1), first_last_orders,
                                         on='client',
                                         how='left')
            first_last_orders = first_last_orders.query("first_order_date == first_order_date")
            self.orders = pd.merge(self.orders, first_last_orders[['client', 'max_order_seq']], on='client', how='left')

            first_last_orders = pd.merge(first_last_orders,
                                         self.orders.query("order_seq_num == 1").groupby(
                                             "client").agg({"payment_amount": 'mean'}).reset_index().rename(
                                             columns={"payment_amount": "first_order_amount"})
                                         , on='client', how='left')

            first_last_orders['download_to_first_order_hourly'] = first_last_orders.apply(
                lambda row: calculate_time_diff(row['download_date'], row['first_order_date'], 'hourly'), axis=1)

            first_last_orders['diff_hours_recency'] = first_last_orders.apply(
                lambda row: calculate_time_diff(row['last_order_date'], max_date, 'hourly'), axis=1)

            x_axis, y_axis = [0, np.mean(first_last_orders['download_to_first_order_hourly'])], [0, np.mean(
                first_last_orders['first_order_amount'])]
            self.orders['download_to_first_order_hourly'] = first_last_orders.apply(
                lambda row: calculate_time_diff(row['download_date'], row['first_order_date'], 'hourly'), axis=1)

            for o in range(1, avg_order_count):  # iterate each order and calculate hour diff. and avg. payment amount.
                _orders = self.orders.query("order_seq_num == @o and next_order_date == next_order_date")
                y_axis.append(np.mean(_orders['payment_amount']))
                x_val = x_axis[-1] + np.mean(list(_orders.query("order_seq_num != 0").groupby("diff_days").agg(
                    {"id": "count"}).reset_index().sort_values(by='id', ascending=False)['diff_days'])[0])
                x_axis.append(x_val)
            # recency value is added as the last point on the x-axis
            x_axis += [x_axis[-1] + np.mean(first_last_orders['diff_hours_recency'])]
            y_axis += [0]
            self.cohorts['customers_journey']['hourly'] = pd.DataFrame(
                zip(x_axis, y_axis)).rename(columns={0: "hourly order differences",
                                                     1: "customers` average Purchase Value"})
        except Exception as e:
            print(e)

    def insert_into_reports_index(self,
                                  cohort,
                                  start_date,
                                  time_period,
                                  _from=0,
                                  _to=1,
                                  cohort_type='orders',
                                  index='orders'):
        """
        via query_es.py, each report can be inserted into the reports index with the given format.
        {"id": unique report id,
         "report_date": start_date or current date,
         "report_name": "cohort",
         "index": "main",
         "report_types": {"time_period": weekly, daily, hourly (only for customers_journey)
                          "type": orders, downloads,
                          "_from": 0 (only for downlods), 1, 2, 3
                          "_to": 1, 2, 3, 4
                          },
         "data": cohort.fillna(0.0).to_dict("results") -  dataframe to list of dictionary
         }
         !!! null values are assigned to 0.

        :param cohort: data set, data frame
        :param start_date: data start date
        :param time_period: daily, weekly
        :param _from: which order is cohort created from?
        :param _to: which order is cohort created to?
        :param cohort_type: orders, downloads, customer_journeys
        :param index: dimensionality of data index orders_location1 ;  dimension = location1
        """
        list_of_obj = [{"id": np.random.randint(200000000),
                        "report_date": current_date_to_day().isoformat() if start_date is None else start_date,
                        "report_name": "cohort",
                        "index": get_index_group(index),
                        "report_types": {"time_period": time_period, "from": _from, "to": _to,  "type": cohort_type},
                        "data": cohort.fillna(0.0).to_dict("results")}]
        self.query_es.insert_data_to_index(list_of_obj, index='reports')

    def get_cohort_name(self, cohort_name):
        """
        cohort_name format; cohort_orders_from_1_to_2_weekly;
        This indicates a weekly cohort related to customers who order their 2nd orders.
        This structured cohort name is split in order to collect _from, _to, _cohort_type filters.
        :param cohort_name: structured cohort name
        :return: _cohort_type, _from, _to (string). These are directly sent to 'fetch' and 'insert_into_reports_index'.
        """
        _cohort_type = cohort_name.split("_")[1]
        _from, _to = 0, 1
        if _cohort_type == 'orders':
            try:
                _from, _to = int(cohort_name.split("_")[3]), int(cohort_name.split("_")[5])
            except Exception as e:
                print("from - to not in cohort_name !!")
        return _cohort_type, _from, _to

    def execute_cohort(self, start_date):
        """
        1.  collect downloads and orders data from given indexes.
        2.  create From Download to 1st Order Cohort per week and day.
        3.  create From 1st/2nd/3rd order To 2nd/3rd/4th order Cohort per week and day.
        4.  create customer average journey From Download to the order count
            that is the average order count of all customers.
        5.  At the end 9 individual reports are created.
            Here are the created reports after the execution;
                -   cohort_downloads_daily
                -   cohort_downloads_weekly
                -   cohort_orders_from_1_to_2_daily
                -   cohort_orders_from_2_to_3_daily
                -   cohort_orders_from_3_to_4_daily
                -   cohort_orders_from_1_to_2_weekly
                -   cohort_orders_from_2_to_3_weekly
                -   cohort_orders_from_3_to_4_weekly
                -   customers_journey_hourly

        6.  insert each cohort individually into 'reports' index.

        :param start_date: starting date of collecting data
        """
        self.get_data(start_date)
        self.cohort_download_to_1st_order()
        self.cohort_from_to_order()
        self.customer_average_journey()
        for _c in self.cohorts:
            if _c != 'customers_journey':
                for p in self.cohorts[_c]:
                    _cohort_type, _from, _to = self.get_cohort_name('cohort_' +_c)
                    try:
                        self.cohorts[_c][p][p] = self.cohorts[_c][p][p].apply(lambda x: str(x)[0:10])
                        self.insert_into_reports_index(self.cohorts[_c][p],
                                                       start_date,
                                                       _from=_from,
                                                       _to=_to,
                                                       time_period=p,
                                                       cohort_type=_cohort_type,
                                                       index=self.order_index)
                    except Exception as e:
                        print(e)

        try:
            self.insert_into_reports_index(self.cohorts['customers_journey']['hourly'],
                                           start_date,
                                           time_period='hourly',
                                           _from=0,
                                           _to=100,
                                           cohort_type='customers_journey',
                                           index=self.order_index)
        except Exception as e:
            print(e)

    def fetch(self, cohort_name, start_date=None, end_date=None):
        """
        Example of cohort_name;

            cohort_orders_from_1_to_2_daily;
                cohort_type; orders
                time_period; daily
                orders from; 1
                orders to; 2

        Directly, these arguments are sent to elasticsearch reports index in order to fetch related reports.

        :param cohort_name: e.g. cohort_orders_from_1_to_2_daily
        :param _from: 1, 2, 3, 4 no need when it is for 'cohort_download_daily' / 'cohort_download_weekly'
        :param _to: 2, 3, 4
        :param start_date: filter cohort date start
        :param end_date: directly sending end_Date to report_date in reports index.
        :return: data data-frame
        """
        _cohort_type, _from, _to = self.get_cohort_name(cohort_name)
        _time_period = cohort_name.split("_")[-1]
        boolean_queries, date_queries = [], []
        boolean_queries = [{"term": {"report_name": "cohort"}},
                           {"term": {"index": get_index_group(self.order_index)}},
                           {"term": {"report_types.time_period": _time_period}},
                           {"term": {"report_types.type": _cohort_type}},
                           {"term": {"report_types.from": _from}},
                           {"term": {"report_types.to": _to}}]

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
            if start_date is not None:
                _data[_time_period] = _data[_time_period].apply(lambda x: convert_to_date(x))
                if _time_period not in ['yearly', 'hourly']:
                    start_date = convert_to_date(start_date)
                    _data = _data[_data[_time_period] >= start_date]
        return _data













