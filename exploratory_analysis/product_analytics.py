import numpy as np
import pandas as pd
import datetime
import random
from time import gmtime, strftime
import pytz
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import argparse

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from configs import default_es_port, default_es_host, elasticsearch_settings
from utils import find_week_of_monday, convert_dt_to_day_str, convert_str_to_hour
from data_storage_configurations.query_es import QueryES


class ProductAnalytics:
    """

    """
    def __init__(self, has_download=True, host=None, port=None):
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.query_es = QueryES(port=port, host=host)
        self.has_download = has_download
        self.download_field_data = ["id", "download_date", "client"]
        self.session_orders_field_data = ["id", "session_start_date", "client"]
        self.downloads = pd.DataFrame()
        self.orders = pd.DataFrame()
        self.sessions = pd.DataFrame()
        self.download_to_first_order = pd.DataFrame()
        self.download_to_first_order_cohort_weekly = pd.DataFrame()
        self.download_to_first_order_cohort_daily = pd.DataFrame()
        self.orders_from_to_daily = pd.DataFrame()
        self.orders_from_to_weekly = pd.DataFrame()