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
from werkzeug.utils import secure_filename

from customeranalytics.utils import abspath_for_sample_data, read_yaml, current_date_to_day, convert_to_date
from customeranalytics.configs import query_path, chart_names, ALLOWED_IMAGE_EXTENSIONS, MAX_IMAGE_FILESIZE, TIME_DIFF_STR, DATE_DIFF_STR
from customeranalytics.web.app.home.forms import SampleData, RealData, Charts, charts

engine = create_engine('sqlite://///' + join(abspath_for_sample_data(), "web", 'db.sqlite3'), convert_unicode=True,
                       connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()

samples = SampleData()
real = RealData()
charts = Charts(samples.kpis, real)


def image_format(filename):
    return filename.rsplit(".", 1)[1]


def allowed_image(filename):

    if not "." in filename:
        return False

    ext = image_format(filename)

    if ext.upper() in ALLOWED_IMAGE_EXTENSIONS:
        return True
    else:
        return False


def allowed_image_filesize(request):
    if "filesize" in request.cookies:
        if int(request.cookies['filesize']) <= MAX_IMAGE_FILESIZE:
            return True
        else:
            return False
    else:
        return True


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
        self.message_sep = "#*#*£½"
        self.message_key_sep = "_1_1_1_"
        self.message_key_val_sep = ":1:1:1:"

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

    def delete_query(self, table, condition):
        _query = "DELETE FROM " + table
        _query += " WHERE " + condition
        _query = _query.replace("\\", "")
        return _query

    def check_for_table_exits(self, table):
        """
        checking sqlite if table is created before. If it not, table is created.
        :params table: checking table name in sqlite
        """
        if table not in list(self.tables['name']):
            con.execute(self.sqlite_queries[table])

    def read_chats(self):
        try:
            return pd.read_sql("""
                                SELECT chat.*, user_avatar.user_avatar
                                FROM chat LEFT JOIN (SELECT user, user_avatar FROM user_avatar) AS user_avatar 
                                ON chat.user = user_avatar.user
            """, con)
        except Exception as e:
            return pd.DataFrame()

    def check_sub_chats(self, messages):
        updated_messages = ""
        if messages != "":
            updated_messages = []
            for m in messages.split(self.message_sep):
                d = {k.split(self.message_key_val_sep)[0]: k.split(self.message_key_val_sep)[1] for k in
                     m.split(self.message_key_sep)}
                d['date_1'] = self.get_time_diff_string(convert_to_date(d['date']))
                d['date_2'] = self.get_date_diff_string(convert_to_date(d['date']))
                d['user_avatar'] = self.fetch_pic(user=d['user'])
                updated_messages.append(d)
        return updated_messages

    def find_user(self):
        try:
            user = current_user.username
        except Exception as e:
            user = "_"
        return user

    def get_plots(self, plots):
        charts_for_profile = {}
        _updated_chart_names = []
        for plot in plots:
            _chart_name = None
            if plot != '':
                _target, _chart, _index, _date = plot.split("*")
                graph_json = charts.get_individual_chart(target=_target, chart=_chart, index=_index, date=_date if _date != '' else None)
                charts_for_profile[_chart] = charts.get_json_format(graph_json)
                _chart_name = _chart
            _updated_chart_names.append(_chart_name)
        for page in chart_names:
            for i in chart_names[page]:
                _chart_name = chart_names[page][i].split("*")[-1]
                if _chart_name not in list(charts_for_profile.keys()):
                    charts_for_profile[chart_names[page][i].split("*")[-1]] = {'trace': [], 'layout': []}
        return charts_for_profile, _updated_chart_names

    def get_time_diff_string(self, date):
        _total_sec = abs(date - current_date_to_day()).total_seconds()
        counter = 0
        detected = False
        _format = ("%b %d %H:%M ", "%p") if _total_sec < (60 * 60 * 24 * 365) else ("'%y %b %d %H:%M ", "%p")
        time_diff_str = ""
        default_condition = lambda x: True if x >= len(TIME_DIFF_STR) - 1 else False
        condition = lambda x: True if TIME_DIFF_STR[x][0] <= _total_sec < TIME_DIFF_STR[x + 1][0] else False
        while not detected:
            if condition(counter) or default_condition(counter):
                value = ""
                try:
                    value = str(int(_total_sec / TIME_DIFF_STR[counter][0]))
                except Exception as e:
                    print(e)
                time_diff_str = value + TIME_DIFF_STR[counter + 1][1]
                detected = True
            counter += 1
        return time_diff_str

    def get_date_diff_string(self, date):
        _total_sec = abs(date - current_date_to_day()).total_seconds()
        _format = ("%b %d %H:%M ", "%p") if _total_sec < (60 * 60 * 24 * 365) else ("'%y %b %d %H:%M ", "%p")
        date_str, prefix_date_str = [""]*2

        for p_d_str in range(2):
            if DATE_DIFF_STR[p_d_str][0] < _total_sec < DATE_DIFF_STR[p_d_str + 1][0]:
                prefix_date_str = DATE_DIFF_STR[p_d_str + 1][1]
        if prefix_date_str in [DATE_DIFF_STR[0][1], DATE_DIFF_STR[1][1]]:
            date_str = DATE_DIFF_STR[0][1] + " " + date.strftime("%b %d %H:%M ") + date.strftime("%p").lower()
            datetime.datetime.now().strftime("%b %d %H:%M ") + datetime.datetime.now().strftime("%p").lower()
        else:
            date_str = date.strftime(_format[0]) + date.strftime(_format[1]).lower()
        return date_str

    def fetch_chats(self):
        recent_chats = self.read_chats()
        charts_for_profiles = {}
        if len(recent_chats) != 0:
            recent_chats['message'] = recent_chats['message'].apply(lambda x: self.check_sub_chats(x))
            recent_chats['date'] = recent_chats['date'].apply(lambda x: convert_to_date(x))
            recent_chats = recent_chats.sort_values('date', ascending=False)
            recent_chats['date_1'] = recent_chats['date'].apply(lambda x: self.get_time_diff_string(x))
            recent_chats['date_2'] = recent_chats['date'].apply(lambda x: self.get_date_diff_string(x))
            charts_for_profiles, recent_chats['chart_name'] = self.get_plots(list(recent_chats['chart']))
        self.filters = {"dimensions": real.get_report_dimensions(), "chart_names": chart_names}
        return {"messages": recent_chats.to_dict('results') if len(recent_chats) != 0 else None,
                'charts': charts_for_profiles, 'filters': self.filters}

    def fetch_pic(self, user=None):
        _user_name = current_user.username
        logo = "info.pic"
        if user is not None:
            _user_name = user
        try:
            logo = list(pd.read_sql("""
                                        SELECT user_avatar 
                                        FROM user_avatar 
                                        WHERE user = '{}'
                                    """.format(_user_name), con)['user_avatar'])[0]
        except Exception as e:
            print(e)
            logo = "info.jpeg"
        return logo

    def add_pic(self, request):
        """

        """

        try:
            image = request.files["image"]
            filename = secure_filename(image.filename)
            if allowed_image(filename) and allowed_image_filesize(request):
                updated_filename = ".".join([current_user.username, image_format(filename)])
                path = os.path.join(abspath_for_sample_data(),
                                    "web", "app", "base", "static", "assets", "img", "avatars", updated_filename)
                image.save(path)

                try:
                    self.check_for_table_exits('user_avatar')
                except Exception as e:
                    print()

                try:
                    con.execute(self.delete_query(table='user_avatar',
                                                  condition=" user = '" + current_user.username + "' "))
                except Exception as e:
                    print()
                try:
                    con.execute(self.insert_query(table='user_avatar',
                                                  columns=self.sqlite_queries['columns']['user_avatar'][1:],
                                                  values={'user': current_user.username,
                                                          "user_avatar": updated_filename}
                                                  ))
                except Exception as e:
                    print()

        except Exception as e:
            print("no files uploaded")

    def add_new_message(self, request):
        """

        """
        if request != {}:
            _user = self.find_user()
            _user_logo = self.fetch_pic()
            try:
                _chart_name = chart_names[request['chart'].split("*")[0]][request['chart'].split("*")[1]]
                _index = request['index']
                _date = '' if request['date'] == '' else request['date']
                _chart = _chart_name + "*" + _index + "*" + _date
            except Exception as e:
                _chart = ""

            try:
                self.check_for_table_exits("chat")
            except Exception as e:
                print()

            _ts = str(current_date_to_day())
            _message = "user" + self.message_key_val_sep +_user + self.message_key_sep + \
                       "user_avatar" + self.message_key_val_sep + _user_logo + self.message_key_sep + \
                       "date" + self.message_key_val_sep + _ts + self.message_key_sep + \
                       "message" + self.message_key_val_sep + request['message']
            _id = request.get('id', None)
            if _id is not None:
                _id = request['id']
                chat = self.read_chats()
                chat = chat[chat['id'] == int(_id)]
                chat['message'] = chat['message'] + self.message_sep + _message

                try:
                    con.execute(self.update_query(table='chat',
                                                  condition=" id = " + str(_id),
                                                  columns=['message'], values={'message': list(chat['message'])[0]}))
                except Exception as e:
                    print()

            else:
                chat = {'user': _user,
                        'general_message': request['general_message'],
                        'message': _message,
                        'date': _ts,
                        'chart': _chart,
                        'chat_type': 'message',
                        'user_logo': _user + '.png'}
                try:
                    con.execute(self.insert_query(table='chat',
                                                  columns=self.sqlite_queries['columns']['chat'][1:],
                                                  values=chat
                                                  ))
                except Exception as e:
                    print()



