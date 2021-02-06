from flask_wtf import FlaskForm
from wtforms import TextField, PasswordField
from wtforms.validators import InputRequired, Email, DataRequired
import pandas as pd
import json

from os.path import abspath, join, dirname, basename
from os import listdir
import plotly.graph_objs as go
import plotly

from utils import convert_to_day
from configs import time_periods

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
            {'orders_monthly': {'trace': go.Scatter(mode="lines+markers+text",
                                                    line=dict(color='firebrick', width=4),
                                                    textposition="bottom center",
                                                    textfont=dict(
                                                        family="sans serif",
                                                        size=30,
                                                        color="crimson")),
                                'layout': go.Layout()},
             'orders_hourly': {'trace': go.Scatter(mode="lines+markers+text",
                                                   line=dict(color='firebrick', width=4),
                                                   textposition="bottom center",
                                                   textfont=dict(
                                                       family="sans serif",
                                                       size=30,
                                                       color="crimson")),
                               'layout': go.Layout()},
             'orders_weekly': {'trace': go.Scatter(mode="lines+markers+text",
                                                   line=dict(color='firebrick', width=4),
                                                   textposition="bottom center",
                                                   textfont=dict(
                                                       family="sans serif",
                                                       size=30,
                                                       color="crimson")),
                               'layout': go.Layout()},
             'orders_daily': {'trace': go.Scatter(mode="lines+markers+text",
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
    }
}


class SampleData:
    """
    This enables us to visualize dashboards when there is no initial data has been created yet.
    These sample data comes from the sample data_folder default in the library.
    """
    kpis = {}
    _base_path = abspath("").split(basename(abspath("")))[0]
    folder = join(_base_path, "exploratory_analysis", 'sample_data', '')
    for f in listdir(dirname(folder)):
        if f.split(".")[1] == 'csv':
            _data = pd.read_csv(join(folder, f))
            if 'date' in list(_data.columns):
                _data['date'] = _data['date'].apply(lambda x: convert_to_day(x))
            kpis["_".join(f.split(".")[0].split("_")[2:])] = _data


class RealData:
    """
    Just, For now, it is the same as Samples. (Work in progress)
    """
    kpis = {}

    ## TODO: data for all kpis will be directly from elasticsearch report index
    _base_path = abspath("").split(basename(abspath("")))[0]
    folder = join(_base_path, "exploratory_analysis", 'sample_data', '')
    for f in listdir(dirname(folder)):
        if f.split(".")[1] == 'csv':
            kpis["_".join(f.split(".")[0].split("_")[2:])] = pd.read_csv(join(folder, f))


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

    def get_data(self, chart):
        """
        checks both sample and real data. If there is real data for the related KPI or chart fetches from sample_data.
        :param chart: e.g. rfm, customer_segmentation, ...
        :return:
        """
        if chart not in list(self.reals.keys()):
            list(self.samples.keys())
            return self.samples[chart]
        else:
            return self.reals[chart]

    def get_trace(self, trace, chart):
        """
        fill more variables on charts dictionary. At this process, data sets are stored in the trace.

        :param trace: e.g. [{go.Scatter()}]
        :param chart: e.g. rfm, customer_segmentation, ...
        :return:
        """
        _data = self.get_data(chart) # collect data
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
        return [trace]

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
            self.graph_json['charts'][c] = json.dumps({'trace': trace,
                                                       'layout': charts[target]['charts'][c]['layout']},
                                                      cls=plotly.utils.PlotlyJSONEncoder)
        # collecting KPIs
        self.graph_json['kpis'] = {}
        for k in charts[target]['kpis']:
            _obj = self.get_values(k)
            for _k in charts[target]['kpis'][k]:
                self.graph_json['kpis'][_k] = '{:,}'.format(int(_obj[_k])).replace(",", ".")
