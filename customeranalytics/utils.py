import datetime
import os
import inspect
from os.path import join
import yaml

from customeranalytics.configs import none_types

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
    if date == date and date not in none_types:
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
    :param directory: path
    :param filename: file name with .yaml format
    :return:
    """
    with open(join(directory, "", filename)) as file:
        docs = yaml.full_load(file)
    return docs


def convert_to_iso_format(date):
    """
    converting iso-format to timestamp Ex: from  2021-01-01T00:00:00 to 2021-01-01 00:00:00
    """
    if len(str(date)) == 10:
        return datetime.datetime.strptime(str(date)[0:10] + ' 00:00:00', "%Y-%m-%d %H:%M:%S").isoformat()
    if len(str(date)) == 19:
        return datetime.datetime.strptime(str(date)[0:10] + ' ' + str(date)[11:19], "%Y-%m-%d %H:%M:%S").isoformat()
    if len(str(date)) != 10 and len(str(date)) != 19:
        return date.isoformat()


def get_index_group(index):
    """
    when Exploratory Analysis or Ml Processes work on dimension this will hep us to get exact dimension name.
    If it is calculating for aLL data, this will return 'main'.
    """
    if index in ['orders', 'downloads']:
        return 'main'
    else:
        return index


def convert_dt_to_month_str(date):
    """
    converting date time to str date. e.g; 2020-12-12 00:00:14  - 2020-12-12.
    :param date: datetime format; %Y-%m-%d %H:%M:%S
    :return: string date
    """
    return datetime.datetime.strptime(str(date)[0:7], "%Y-%m")


def sqlite_string_converter(_str, back_to_normal=False):
    """
    when query or path is inserted into the sqlite db, it is need to be convert ''' and removing back slashes.
    :param _str: query string Ex: " SELECT *  FROm table ..."
    :param back_to_normal: convert back to normal True / False
    :return: string query
    """
    if back_to_normal:
        return _str.replace("#&_5", "'").replace("+", " ") + ' '
    else:
        return _str.replace("'", "#&_5").replace("\r", " ").replace("\n", " ").replace(" ", "+")


def abspath_for_sample_data():
    """
    get customer_analytics path. Ex: ....../customer_analytics
    :return: current folder path
    """
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    base_name = os.path.basename(currentdir)
    while base_name != 'customeranalytics':
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        base_name = os.path.basename(currentdir)
    return currentdir


def dimension_decision(order_index):
    """
    Decision of dimension.
    if order_index = 'Orders' it will be whole data, if it is not it will be executed for a dimension
    :param order_index:
    :return: True/False
    """
    if order_index != 'orders':
        return True
    else:
        return False


def formating_numbers(num):
    """
    Human formatting for long numbers as strings
    """
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if num % 1 == 0:
        num = int(num)
        return num + ['', 'K', 'M', 'G', 'T', 'P'][magnitude]

    else:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
