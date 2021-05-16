import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pandas as pd
from numpy import array
import json
import datetime

from os.path import join, dirname, exists
from flask_login import current_user
from os import listdir
import plotly.graph_objs as go
import plotly
from screeninfo import get_monitors
from sqlalchemy import create_engine, MetaData

from utils import convert_to_day, abspath_for_sample_data, read_yaml, convert_to_date, current_date_to_day
from configs import time_periods, descriptive_stats, abtest_promotions, abtest_products, abtest_segments, query_path, messages, chart_names
from web.app.home.forms import SampleData, RealData, Charts, charts

engine = create_engine('sqlite://///' + join(abspath_for_sample_data(), "web", 'db.sqlite3'), convert_unicode=True,
                       connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()

samples = SampleData()
real = RealData()
charts = Charts(samples.kpis, real)


class Profiles:
    def __init__(self, user=None):
        self.user = user
        self.users = []
        self.sqlite_queries = read_yaml(query_path, "queries.yaml")
        self.tables = pd.read_sql(self.sqlite_queries['tables'], con)
        self.data_types = {'orders': 'Sessions', 'downloads': 'Customers', 'products': 'Products (Baskets)'}
        self.log_file = []
        self.recent_chats = pd.DataFrame()
        self.filters = ['main']

    def insert_query(self, table, columns, values):
        values = [values[col] for col in columns]
        _query = "INSERT INTO " + table + " "
        _query += " (" + ", ".join(columns) + ") "
        _query += " VALUES (" + ", ".join([" '{}' ".format(v) for v in values]) + ") "
        _query = _query.replace("\\", "")
        return _query

    def update_query(self, table, condition, columns, values):
        values = [(col, values[col]) for col in columns if values.get(col, None) is not None]
        _query = "UPDATE " + table
        _query += " SET " + ", ".join([i[0] + " = '" + i[1] + "'" for i in values])
        _query +=" WHERE " + condition
        _query = _query.replace("\\", "")
        return _query

    def update_profiles(self):
        """

        """

    def create_notifications(self, user, text, ts):
        splits = text.split("-")
        if splits[0] == 'ds':
            message = messages["ds_connect"].format(self.data_types[splits[1].split("_")[0]])
        else:
            message = messages["_".join([splits[0], splits[-1]])]

        chat = {'user': user,
                'general_message': message,
                'message': "",
                'date': ts,
                'chart': None,
                'chat_type': 'info',
                'user_logo': user + '.png'}
        return chat

    def collect_logs(self, last_n_rows=200):
        """

        """
        log_file = []
        try:
            es_tag = pd.read_sql("SELECT * FROM es_connection", con).to_dict('results')[-1]
            if exists(join(es_tag['directory'], "logs.log")):
                file_path = join(es_tag['directory'], "logs.log")
                with open(file_path) as f:
                    self.log_file = f.readlines()
            log_file =log_file[-min(len(log_file), last_n_rows):]
        except Exception as e:
                print(e)
        return log_file

    def read_chats(self):
        try:
            return pd.read_sql("select * from chat WHERE user = '{}'".format(self.user), con)
        except Exception as e:
            return pd.DataFrame()

    def read_users(self):
        users = []
        try:
            users = list(pd.read_sql("select username from user", con)['username'])
        except Exception as e:
            print(e)
        return users

    def create_chats(self, log_file, users):
        recent_chats = []
        for l in log_file:
            if l != '\n':
                try:
                    ts = l[0:19]
                    if convert_to_date(ts) > current_date_to_day() - datetime.timedelta(days=7):
                        for user in users:
                            if user in l:
                                print(l)
                                recent_chats.append(self.create_notifications(user, l.split(self.user)[-1][3:].replace("\n", ""), ts))
                except Exception as e:
                    print(e, l)
        recent_chats = pd.DataFrame(recent_chats).reset_index()
        _chats = self.read_chats()

        if len(_chats) != 0:
            removing_recent_chats = list(
                pd.merge(_chats[['user', 'epoch']], self.recent_chats[['user', 'epoch', 'index']], on=['user', 'epoch'],
                         how='inner')['index'])
            if len(removing_recent_chats) != 0:
                recent_chats = recent_chats.query("index not in @removing_recent_chats")
            recent_chats = pd.concat([_chats, recent_chats])
        return recent_chats

    def check_sub_chats(self, r_chats):
        if len() != 0:
            r_chats['message'] = r_chats['message'].apply(lambda x: x.split("#*#*£½"))
        return r_chats

    def find_user(self):
        try:
            user = current_user.username
        except Exception as e:
            user = "_"
        return user

    def fetch_chats(self):
        #user = self.find_user()
        #users = self.read_users()
        #logs = self.collect_logs()
        #recent_chats = self.create_chats(log_file=logs, users=users)
        #recent_chats = self.check_sub_chats(r_chats=recent_chats)
#
        ## get filters
        self.filters = {"dimensions": real.get_report_dimensions(), "chart_names": chart_names}

        return {"messages": None, #recent_chats.to_dict('results'),
         'filters': self.filters}

    def add_new_message(self, request):
        """

        """
        if request != {}:
            _chart_name = chart_names[request['chart'].split("*")[0]][request['chart'].split("*")[1]]
            _index = request['index']
            _date = request['date']
            _user = self.find_user()
            _ts = str(current_date_to_day())
            chat = {'user': _user,
                    'general_message': request['general_message'],
                    'message': request['message'],
                    'date': _ts,
                    'chart': "*".join([_chart_name, _index, _date]),
                    'chat_type': 'message',
                    'user_logo': _user + '.png'}

            print(chat)

            # charts.get_individual_chart(target=_target, chart=_chart, index=_index, date=_date)
            print()

