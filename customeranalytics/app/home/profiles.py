import pandas as pd
import datetime
import os

from flask_login import current_user
from werkzeug.utils import secure_filename

from customeranalytics.data_storage_configurations import DataStorageConfigurations
from customeranalytics.app.home.forms import Charts


def image_format(filename):
    return filename.rsplit(".", 1)[1]


def allowed_image(filename):

    if not "." in filename:
        return False

    ext = image_format(filename)

    if ext.upper() in DataStorageConfigurations.ALLOWED_IMAGE_EXTENSIONS:
        return True
    else:
        return False


def allowed_image_filesize(request):
    if "filesize" in request.cookies:
        if int(request.cookies['filesize']) <= DataStorageConfigurations.MAX_IMAGE_FILESIZE:
            return True
        else:
            return False
    else:
        return True


class Profiles(DataStorageConfigurations, Charts):
    def __init__(self, user=None):
        super().__init__()
        self.user = user
        self.users = []
        self.tables = pd.read_sql(self.sqlite_queries['tables'], self.con)
        self.data_types = {'orders': 'Sessions', 'downloads': 'Customers', 'products': 'Products (Baskets)'}
        self.log_file = []
        self.recent_chats = pd.DataFrame()
        self.filters = ['main']
        self.message_sep = "#*#*£½"
        self.message_key_sep = "_1_1_1_"
        self.message_key_val_sep = ":1:1:1:"
        self.charts_query = """
            SELECT chat.*, user_avatar.user_avatar
            FROM chat LEFT JOIN (SELECT user, user_avatar FROM user_avatar) AS user_avatar 
            ON chat.user = user_avatar.user
        """
        self.user_logo_pic_query = f"""
            SELECT user_avatar 
            FROM user_avatar 
            WHERE user = '{current_user.username}'
        """

    def check_for_table_exits(self, table):
        """
        checking sqlite if table is created before. If it not, table is created.
        :params table: checking table name in sqlite
        """
        if table not in list(self.tables['name']):
            self.con.execute(self.sqlite_queries[table])

    def check_sub_chats(self, messages):
        updated_messages = ""
        if messages != "":
            updated_messages = []
            for m in messages.split(self.message_sep):
                d = {k.split(self.message_key_val_sep)[0]: k.split(self.message_key_val_sep)[1] for k in
                     m.split(self.message_key_sep)}
                d['date_1'] = self.get_time_diff_string(self.convert_to_date(d['date']))
                d['date_2'] = self.get_date_diff_string(self.convert_to_date(d['date']))
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
                graph_json = self.get_individual_chart(
                    target=_target,
                    chart=_chart,
                    index=_index,
                    date=_date if _date != '' else None
                )
                charts_for_profile[_chart] = self.get_json_format(graph_json)
                _chart_name = _chart
            _updated_chart_names.append(_chart_name)
        for page in self.chart_names:
            for i in self.chart_names[page]:
                _chart_name = self.chart_names[page][i].split("*")[-1]
                if _chart_name not in list(charts_for_profile.keys()):
                    charts_for_profile[self.chart_names[page][i].split("*")[-1]] = {'trace': [], 'layout': []}
        return charts_for_profile, _updated_chart_names

    def get_time_diff_string(self, date):
        _total_sec = abs(date - self.current_date_to_day()).total_seconds()
        counter = 0
        detected = False
        _format = ("%b %d %H:%M ", "%p") if _total_sec < (60 * 60 * 24 * 365) else ("'%y %b %d %H:%M ", "%p")
        time_diff_str = ""
        default_condition = lambda x: True if x >= len(self.TIME_DIFF_STR) - 1 else False
        condition = lambda x: True if self.TIME_DIFF_STR[x][0] <= _total_sec < self.TIME_DIFF_STR[x + 1][0] else False
        while not detected:
            if condition(counter) or default_condition(counter):
                value = ""
                try:
                    value = str(int(_total_sec / self.TIME_DIFF_STR[counter][0]))
                except Exception as e:
                    print(e)
                time_diff_str = value + self.TIME_DIFF_STR[counter + 1][1]
                detected = True
            counter += 1
        return time_diff_str

    def get_date_diff_string(self, date):
        _total_sec = abs(date - self.current_date_to_day()).total_seconds()
        _format = ("%b %d %H:%M ", "%p") if _total_sec < (60 * 60 * 24 * 365) else ("'%y %b %d %H:%M ", "%p")
        date_str, prefix_date_str = [""]*2

        for p_d_str in range(2):
            if self.DATE_DIFF_STR[p_d_str][0] < _total_sec < self.DATE_DIFF_STR[p_d_str + 1][0]:
                prefix_date_str = self.DATE_DIFF_STR[p_d_str + 1][1]
        if prefix_date_str in [self.DATE_DIFF_STR[0][1], self.DATE_DIFF_STR[1][1]]:
            date_str = self.DATE_DIFF_STR[0][1] + " " + date.strftime("%b %d %H:%M ") + date.strftime("%p").lower()
            datetime.datetime.now().strftime("%b %d %H:%M ") + datetime.datetime.now().strftime("%p").lower()
        else:
            date_str = date.strftime(_format[0]) + date.strftime(_format[1]).lower()
        return date_str

    def fetch_chats(self):
        recent_chats = self.read_query(self.charts_query)
        charts_for_profiles = {}
        if len(recent_chats) != 0:
            recent_chats['message'] = recent_chats['message'].apply(self.check_sub_chats)
            recent_chats['date'] = recent_chats['date'].apply(self.convert_to_date)
            recent_chats.sort_values('date', ascending=False, inplace=True)
            recent_chats['date_1'] = recent_chats['date'].apply(self.get_time_diff_string)
            recent_chats['date_2'] = recent_chats['date'].apply(self.get_date_diff_string)
            charts_for_profiles, recent_chats['chart_name'] = self.get_plots(list(recent_chats['chart']))
        self.filters = {"dimensions": self.get_report_dimensions(), "chart_names": self.chart_names}
        return {
            "messages": recent_chats.to_dict('results') if len(recent_chats) != 0 else None,
            'charts': charts_for_profiles,
            'filters': self.filters
        }

    def fetch_pic(self, user=None):
        _user_name = current_user.username
        logo = "info.pic"
        if user is not None:
            _user_name = user
        try:
            logo = list(self.read_query(self.user_logo_pic_query)['user_avatar'])[0]
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
                path = os.path.join(self.abspath_for_sample_data(),
                                    "web", "app", "base", "static", "assets", "img", "avatars", updated_filename)
                image.save(path)

                try:
                    self.check_for_table_exits('user_avatar')
                except Exception as e:
                    print()

                try:
                    self.con.execute(self.delete_query(table='user_avatar',
                                                  condition=" user = '" + current_user.username + "' "))
                except Exception as e:
                    print()
                try:
                    self.con.execute(self.insert_query(table='user_avatar',
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
                _chart_name = self.chart_names[request['chart'].split("*")[0]][request['chart'].split("*")[1]]
                _index = request['index']
                _date = '' if request['date'] == '' else request['date']
                _chart = _chart_name + "*" + _index + "*" + _date
            except Exception as e:
                _chart = ""

            try:
                self.check_for_table_exits("chat")
            except Exception as e:
                print()

            _ts = str(self.current_date_to_day())
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
                    self.con.execute(self.update_query(table='chat',
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
                    self.con.execute(self.insert_query(table='chat',
                                                  columns=self.sqlite_queries['columns']['chat'][1:],
                                                  values=chat
                                                  ))
                except Exception as e:
                    print()



