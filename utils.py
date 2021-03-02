import datetime
from os.path import join
import yaml


def current_date_to_day():
    """
    recent date of converting to datetime from isoformat.
    :return: datetime
    """
    return datetime.datetime.strptime(str(datetime.datetime.now())[0:19], '%Y-%m-%d %H:%M:%S')


def find_week_of_monday(date):
    """
    Monday of each week. This helps us to aggregate data per week.
    :param date: datetime format; %Y-%m-%d %H:%M:%S
    :return: datetime
    """
    week_day = datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d").isoweekday()
    return datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d") - datetime.timedelta(days=week_day-1)


def convert_dt_to_day_str(date):
    """
    converting date time to str date. e.g; 2020-12-12 00:00:14  - 2020-12-12.
    :param date: datetime format; %Y-%m-%d %H:%M:%S
    :return: string date
    """
    return datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d")


def convert_str_to_hour(date):
    """
    string date is converting to datetime and grabbing hour from it.
    :param date: datetime format; %Y-%m-%d %H:%M:%S
    :return: hour [0 - 24)
    """
    return datetime.datetime.strptime(str(date)[0:10] + ' ' + str(date)[11:19], "%Y-%m-%d %H:%M:%S").hour


def convert_to_date(date):
    """
    Convert str date to datetime. If is NULL skip the process and return None.
    :param date: str format; %Y-%m-%d %H:%M:%S
    :return:
    """
    if date == date:
        return datetime.datetime.strptime(str(date)[0:10] + ' ' + str(date)[11:19], "%Y-%m-%d %H:%M:%S")
    else:
        return None


def calculate_time_diff(date1, date2, period):
    """
        -   Calculating total seconds between date1 (start date) and date2 (end date).
        -   Calculated total seconds is divided to time period.
    :param date1: start date
    :param date2: end date
    :param period: period of time that can be measured a countable number of times between start date and end date.
    :return: difference float
    """
    diff = 0
    division = (60 * 60)
    if date1 == date1 and date2 == date2:
        if period == 'hour':
            division = (60 * 60)
        if period == 'day':
            division = (60 * 60 * 24)
        if period == 'week':
            division = (60 * 60 * 24 * 7)
        diff = int(abs(date1 - date2).total_seconds() / division)
    return diff


def convert_to_day(date):
    """
    Convert str date to day.
    :param date: str format; %Y-%m-%d
    :return:
    """
    return datetime.datetime.strptime(str(date)[0:10], "%Y-%m-%d")


def read_yaml(directory, filename):
    """
    reading yaml file with given directory
    :param directory:
    :param filename:
    :return:
    """
    with open(join(directory, "", filename)) as file:
        docs = yaml.full_load(file)
    return docs


def convert_to_iso_format(date):
    if len(str(date)) == 10:
        return datetime.datetime.strptime(str(date)[0:10] + ' 00:00:00', "%Y-%m-%d %H:%M:%S").isoformat()
    if len(str(date)) == 19:
        return datetime.datetime.strptime(str(date)[0:10] + ' ' + str(date)[11:19], "%Y-%m-%d %H:%M:%S").isoformat()


def get_index_group(index):
    _index = index.split("_")
    if len(_index) == 1:
        return 'main'
    else:
        return index.split("_")[-1]


def convert_dt_to_month_str(date):
    """
    converting date time to str date. e.g; 2020-12-12 00:00:14  - 2020-12-12.
    :param date: datetime format; %Y-%m-%d %H:%M:%S
    :return: string date
    """
    return datetime.datetime.strptime(str(date)[0:7], "%Y-%m")


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

