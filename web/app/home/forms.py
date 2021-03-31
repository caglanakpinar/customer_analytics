from flask_wtf import FlaskForm
from wtforms import TextField, PasswordField
from wtforms.validators import InputRequired, Email, DataRequired
import pandas as pd
import json

from os.path import abspath, join, dirname, basename
from os import listdir
import plotly.graph_objs as go
import plotly
from screeninfo import get_monitors
from sqlalchemy import create_engine, MetaData

from utils import convert_to_day
from configs import time_periods
from data_storage_configurations import collect_reports

engine = create_engine('sqlite://///' + join(abspath(""), "web", 'db.sqlite3'), convert_unicode=True, connect_args={'check_same_thread': False})
metadata = MetaData(bind=engine)
con = engine.connect()


"""
charts Dictionary:
    each chart related to each page is stored in chats.
    When need to be visualize, related page of chart is merged with stored data on Charts class
"""
charts = {
    # index.html charts and KPIs
    "index": {
        # charts - orders_monthly, orders_hourly, orders_weekly, orders_daily, customer_segmentation
        "charts":
            {'monthly_orders': {'trace': go.Scatter(mode="lines+markers+text",
                                                    line=dict(color='firebrick', width=4),
                                                    textposition="bottom center",
                                                    textfont=dict(
                                                        family="sans serif",
                                                        size=30,
                                                        color="crimson")),
                                'layout': go.Layout()},
             'hourly_orders': {'trace': go.Scatter(mode="lines+markers+text",
                                                   line=dict(color='firebrick', width=4),
                                                   textposition="bottom center",
                                                   textfont=dict(
                                                       family="sans serif",
                                                       size=30,
                                                       color="crimson")),
                               'layout': go.Layout()},
             'weekly_orders': {'trace': go.Scatter(mode="lines+markers+text",
                                                   line=dict(color='firebrick', width=4),
                                                   textposition="bottom center",
                                                   textfont=dict(
                                                       family="sans serif",
                                                       size=30,
                                                       color="crimson")),
                               'layout': go.Layout()},
             'daily_orders': {'trace': go.Scatter(mode="lines+markers+text",
                                                  line=dict(color='firebrick', width=4),
                                                  textposition="bottom center",
                                                  textfont=dict(
                                                      family="sans serif",
                                                      size=30,
                                                      color="crimson")),
                              'layout': go.Layout()},
             'customer_segmentation': {'trace': go.Treemap(
                 marker_colorscale='Blues',
                 textinfo="label+value+percent parent+percent entry", parents=[""] * 7),
                 'layout': go.Layout(width=1000,
                                     margin=dict(l=1, r=1, t=1, b=1),
                                     height=400)}},
        # KPIs - total_orders, total_visits, total_unique_visitors. index.html left top
        "kpis": {"kpis": ['total_orders', 'total_visits', 'total_unique_visitors',
                          'total_unique_ordered_clients', 'total_revenue', 'total_discount']}},
    # index2.html of Charts and KPIs
    "index2": {
        # charts - rfm
        "charts": {"rfm": {"trace": go.Scatter3d(mode='markers',
                                                 marker=dict(color=None,
                                                             size=12,
                                                             colorscale='Viridis',
                                                             opacity=0.8)),
                           "layout": go.Layout(width=1000,
                                               margin=dict(l=1, r=1, t=1, b=1),
                                               height=600,
                                               scene=dict(
                                                   xaxis_title='recency',
                                                   yaxis_title='monetary',
                                                   zaxis_title='frequency'))
                           }
                   },
        # not any recent KPIs for now
        "kpis": {}
    },
    "funnel": {
        # charts - rfm
        "charts": {_f: {'trace': go.Scatter(mode="lines+markers+text",
                                                        line=dict(color='firebrick', width=4),
                                                        textposition="bottom center",
                                                        textfont=dict(
                                                            family="sans serif",
                                                            size=30,
                                                            color="crimson")),
                                      'layout': go.Layout()}
                   for _f in ['daily_funnel', 'hourly_funnel', 'weekly_funnel', 'monthly_funnel',
                              'daily_funnel_downloads', 'hourly_funnel_downloads',
                              'weekly_funnel_downloads', 'monthly_funnel_downloads']},
        # not any recent KPIs for now
        "kpis": {}
    }
}


class SampleData:
    """
    This enables us to visualize dashboards when there is no initial data has been created yet.
    These sample data comes from the sample data_folder default in the library.
    """
    kpis = {}
    folder = join(abspath(""), "exploratory_analysis", 'sample_data', '')
    for f in listdir(dirname(folder)):
        if f.split(".")[1] == 'csv':
            kpis["_".join(f.split(".")[0].split("_")[2:])] = None

    _base_path = abspath("").split(basename(abspath("")))[0]
    _base_path = abspath("") #.split(basename(abspath("")))[0]
    folder = join(abspath(""), "exploratory_analysis", 'sample_data', '')
    for f in listdir(dirname(folder)):
        if f.split(".")[1] == 'csv':
            _kpi_name = "_".join(f.split(".")[0].split("_")[2:])

            _data = pd.read_csv(join(folder, f))
            if 'date' in list(_data.columns):
                _data['date'] = _data['date'].apply(lambda x: convert_to_day(x))
            kpis["_".join(f.split(".")[0].split("_")[2:])] = _data


class RealData:
    """
    Just, For now, it is the same as Samples. (Work in progress)
    """
    def __init__(self):
        self.kpis = {}

    def connections(self):
        tag = pd.read_sql("SELECT * FROM schedule_data WHERE status = 'on' ", con).to_dict('resutls')[-1]
        es_tag = pd.read_sql("SELECT * FROM es_connection WHERE tag = '" + tag['tag'] + "' ", con).to_dict('resutls')[-1]
        conn = pd.read_sql(
            """
            SELECT  * FROM data_connection  WHERE tag =  '""" + tag['tag'] + "' and process = 'connected'", con)
        conn = conn.query("is_action == 'None' and is_product == 'None' and is_promotion == 'None'")

        return tag, es_tag, conn.to_dict('results')

    def get_index_names(self, c):
        index = 'main'
        if c['dimension'] != 'None':
            index = c['orders_data_source_tag']
        return index

    def report_name(self, k, c):
        if k['report_name'] == 'funnel':
            r_name = k['time_period'] + '_funnel'
            if k['type'] == 'downloads':
                r_name += '_downloads'
        if k['report_name'] == 'cohort':
            r_name = "_".join(['cohort', k['from'], 'to', k['to'], k['time_period']])
        if k['report_name'] == 'stats':
            r_name = k['type']
        if k['report_name'] not in ['cohort', 'funnel', 'stats']:
            r_name = k
        if c['dimension'] != 'None':
            r_name = "_".join([r_name, c['orders_data_source_tag']])
        return r_name

    def execute_real_data(self):
        try:
            tag, es_tag, conn = self.connections()
            if tag['is_exploratory'] != 'None' or tag['is_mlworks'] != 'None':
                for c in conn:
                    index = self.get_index_names(c)
                    reports = collect_reports(es_tag['port'], es_tag['host'], index, tag['max_date_of_order_data'])

                    for k in reports.to_dict('results'):
                        r_name = self.report_name(k, c)
                        self.kpis[r_name] = pd.DataFrame(k['data'])
        except Exception as e:
            print(e)

        return self.kpis


class Charts:
    """
    Collecting Charts for Dashboards;
        There are 2 types of data sets for Charts;
            sample_data; these are built-in .csv file in exploratory_analysis/sample_data folder
            real_data; created from exploratory_analysis reports and ml_processes after creating a real data connection
        After collecting the real data by using **charts** dictionary,
        template of the chart related to page (.html) and its data merge and sending to render_template.
    """
    def __init__(self, samples, reals):
        """
        The main perspective here to store each chart serialized .json file into the self.graph_json
        e.g. index.html has customer_segmentation chart
            self.graph_json['charts']['customer_segmentation'] = {'trace': [], 'layout': {}}
        If each chart needs any specific change pls check **charts** dictionary.
        :param samples: built-in data sets in .csv format, converted to pandas data-frame
        :param reals: created ports in .csv format, converted to pandas data-frame
        """
        self.samples = samples
        self.reals = reals
        self.graph_json = {}
        self.monitor = get_monitors()[0]

    def get_data(self, chart):
        """
        checks both sample and real data. If there is real data for the related KPI or chart fetches from sample_data.
        :param chart: e.g. rfm, customer_segmentation, ...
        :return:
        """
        try:
            if chart not in list(self.reals.keys()):
                return self.samples[chart]
            else:
                return self.reals[chart]
        except Exception as e:
            print()
            return self.samples[chart]


    def get_widths_heights(self, target, chart):
        if chart == 'customer_segmentation':
            if self.monitor.height == 1080 and self.monitor.width == 1920:
                width, height = 720, 450
            if self.monitor.height == 1050 and self.monitor.width == 1680:
                width, height = 1000, 400
        if chart == 'rfm':
            if self.monitor.height == 1080 and self.monitor.width == 1920:
                width, height = 700, 600
            if self.monitor.height == 1050 and self.monitor.width == 1680:
                width, height = 1000, 600
        if chart in ['customer_segmentation', 'rfm']:
            charts[target]['charts'][chart]['layout']['width'] = width
            charts[target]['charts'][chart]['layout']['height'] = height

    def get_trace(self, trace, chart):
        """
        fill more variables on charts dictionary. At this process, data sets are stored in the trace.

        :param trace: e.g. [{go.Scatter()}]
        :param chart: e.g. rfm, customer_segmentation, ...
        :return:
        """
        _data = self.get_data(chart)  # collect data
        # data for line chart daily(sum), weekly(sum), houry(average), monthly(sum)
        if chart in ["_".join(['orders', t]) for t in time_periods]:
            _data = _data.sort_values(by=chart.split("_")[1], ascending=True)
            trace['x'] = list(_data[chart.split("_")[1]])
            trace['y'] = list(_data['orders'])
            if chart.split("_")[1] not in ['daily', 'weekly']:
                trace['text'] = list(_data['orders'])
        # from the customer segmentation Human readable segments with their sizes
        if chart == 'customer_segmentation':
            trace['labels'] = list(_data['segments'])
            trace['values'] = list(_data['value'])
        # sampled clients of numerical recency, monetary and frequency values
        if chart == 'rfm':
            trace['x'] = list(_data['recency'])
            trace['y'] = list(_data['monetary'])
            trace['z'] = list(_data['frequency'])
            trace['marker']['color'] = list(_data['segments_numeric']) # segments are numerical values.
        if 'funnel' in chart.split("_"):
            _tp = list(set(list(_data.columns)) & set(time_periods))[0]
            _trc = trace
            trace = []
            _trc['x'] = list(_data[_tp])
            for _a in set(list(_data.columns)) - set(time_periods):
                _trc['y'] = list(_data[_a])
                trace.append(_trc)
        return [trace] if 'funnel' in chart.split("_") else trace

    def get_values(self, kpi):
        """
        get data related KPI
        :param kpi: .e.g.total_orders, total_visitors, ...
        :return: dictionary with KPIs in keys
        """
        _data = self.get_data(kpi).to_dict('results')[0]
        return _data

    def get_chart(self, target):
        """
        related to 'target', charts and KPIs are collected in order to show on-page.
        The main aim here, fill the self.graph_json dictionary with serialized dictionaries.
        :param target:
        :return:
        """
        # collecting charts
        self.graph_json['charts'] = {}
        for c in charts[target]['charts']:
            trace = self.get_trace(charts[target]['charts'][c]['trace'], c)
            self.get_widths_heights(chart=c, target=target)
            self.graph_json['charts'][c] = json.dumps({'trace': trace,
                                                       'layout': charts[target]['charts'][c]['layout']},
                                                      cls=plotly.utils.PlotlyJSONEncoder)
        # collecting KPIs
        self.graph_json['kpis'] = {}
        for k in charts[target]['kpis']:
            _obj = self.get_values(k)
            for _k in charts[target]['kpis'][k]:
                self.graph_json['kpis'][_k] = '{:,}'.format(int(_obj[_k])).replace(",", ".")
