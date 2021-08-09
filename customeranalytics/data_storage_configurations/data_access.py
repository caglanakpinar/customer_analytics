import pandas as pd
from dateutil.parser import parse
from os.path import join, dirname
from os import listdir
import sys, os, inspect
from os.path import abspath
import glob

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_dask_partitions


class GetData:
    """
    Collecting data for storing Into Orders & Downloads Index
    """
    def __init__(self,
                 config=None,
                 data_source=None,
                 date=None,
                 data_query_path=None,
                 test=None,
                 dask=False):
        self.data_source = data_source if data_source is not None else 'csv'
        self.data_query_path = data_query_path
        self.dask = dask
        self.conn = None
        self.data = pd.DataFrame()
        self.nrows = test
        self.date = date
        self.n_partitions = default_dask_partitions
        self.config = config

    def get_connection(self):
        if self.data_source in ['postgresql', 'awsredshift', 'mysql']:
            server, db, user, pw, port = str(self.config['host']), str(self.config['db']), \
                                         str(self.config['user']), str(self.config['password']),\
                                         int(self.config['port'])
        if self.data_source == 'mysql':
            from mysql import connector
            self.conn = connector.connect(host=server, database=db, user=user, password=pw)
        if self.data_source in ['postgresql', 'awsredshift']:
            import psycopg2
            self.conn = psycopg2.connect(user=user, password=pw, host=server, port=port, database=db)
        if self.data_source == 'googlebigquery':
            from google.cloud.bigquery.client import Client
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config['db']
            self.conn = Client()
        print("db connection is done!")

    def read_csv(self, f, partition=False):
        _data = pd.DataFrame()
        try:
            for sep in [',', ';', ':', '|']:
                print(sep)
                _data = pd.read_csv(filepath_or_buffer=f,
                                    error_bad_lines=False,
                                    encoding="ISO-8859-1",
                                    sep=sep,
                                    nrows=self.nrows)
                if partition:
                    _data = pd.read_csv(filepath_or_buffer=f,
                                        error_bad_lines=False,
                                        encoding="ISO-8859-1",
                                        sep=sep,
                                        nrows=self.nrows, index_col=None, header=0)
                print(_data.head())
                print(len(_data.columns))
                if len(_data.columns) >= 2:
                    break
        except Exception as e:
            print(e)
            print("yeeee")
        return _data

    def query_data_source(self):
        # import data via pandas
        if self.data_source in ['mysql', 'postgresql', 'awsredshift']:
            self.get_connection()
            counter = 0
            try_count = 10 if self.nrows else 100
            while counter < try_count:
                try:
                    self.data = pd.read_sql(self.data_query_path + " LIMIT " + str(self.nrows) if self.nrows else self.data_query_path, self.conn)
                except Exception as e:
                    print(e)
                if len(self.data) > 0:
                    counter = try_count
                counter += 1
        # import data via google
        if self.data_source == 'googlebigquery':
            self.get_connection()
            self.data = self.conn.query(self.data_query_path + " LIMIT " + str(self.nrows) if self.nrows else self.data_query_path).to_dataframe()

        # import via pandas
        if self.data_source == 'csv':
            li = []
            if self.data_query_path.split(".")[-1] != 'csv':
                _files = glob.glob(join(self.data_query_path, "*.csv"))
                if self.nrows:
                    self.data = self.read_csv(self.data_query_path)
                else:
                    for f in _files:
                        li.append(self.read_csv(f, partition=True))
                    self.data = pd.concat(li, axis=0, ignore_index=True)
            else:
                self.data = self.read_csv(self.data_query_path)

        # import data via pyarrow
        if self.data_source == 'parquet':
            if self.data_query_path.split(".")[-1] == 'parquet':
                if self.nrows:
                    self.data = pd.read_parquet(self.data_query_path).iloc[:self.nrows]
                else:
                    self.data = pd.read_parquet(self.data_query_path)
            else:
                _files = glob.glob(join(self.data_query_path, "*.parquet"))
                if self.nrows:
                    self.data = pd.read_parquet(_files[0]).iloc[:self.nrows]
                else:
                    self.data = pd.read_parquet(_files)

        if self.data_source == 'hdf5':
            if self.data_query_path.split(".")[-1] == 'hdf5':
                if self.nrows:
                    self.data = pd.read_hdf(self.data_query_path).iloc[:self.nrows]
                else:
                    self.data = pd.read_hdf(self.data_query_path)
            else:
                _files = glob.glob(join(self.data_query_path, "*.h5"))
                if self.nrows:
                    self.data = pd.read_hdf(_files[0]).iloc[:self.nrows]
                else:
                    self.data = pd.read_hdf(_files)

    def n_partitions_decision(self):
        if self.data_source == 'dask':
            total_usage_mb = sum(self.data.memory_usage()) / 100000000
            if total_usage_mb < 2:
                self.n_partitions = 3
            else: self.n_partitions = int(total_usage_mb) + 1

    def use_dask(self):
        self.n_partitions_decision()
        if self.dask:
            if self.nrows is not None:
                self.data = dd.from_pandas(self.data, npartitions=self.n_partitions)

    def data_execute(self):
        self.query_data_source()
        self.use_dask()











