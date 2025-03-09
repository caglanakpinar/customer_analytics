import pandas as pd
import datetime

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


class Profiles(Charts):
    def __init__(self):
        super().__init__()
        self.users = []
        self.data_types = {'orders': 'Sessions', 'downloads': 'Customers', 'products': 'Products (Baskets)'}
        self.log_file = []
        self.recent_chats = pd.DataFrame()
        self.filters = ['main']
        self.message_sep = "#*#*£½"
        self.message_key_sep = "_1_1_1_"
        self.message_key_val_sep = ":1:1:1:"

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
