import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from flask_wtf import FlaskForm
from wtforms import TextField, PasswordField
from wtforms.validators import InputRequired, Email, DataRequired
import pandas as pd
from numpy import array
import json

from os.path import abspath, join, dirname, basename
from os import listdir
import plotly.graph_objs as go
import plotly
from screeninfo import get_monitors
from sqlalchemy import create_engine, MetaData

from utils import convert_to_day, abspath_for_sample_data
from configs import time_periods
from data_storage_configurations import collect_reports

engine = create_engine('sqlite://///' + join(abspath_for_sample_data(), "web", 'db.sqlite3'), convert_unicode=True,
                       connect_args={'check_same_thread': False})
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
        # charts - Funnel Daily, Monthly, Weekly
        "charts": {_f: {'trace': go.Scatter(mode="lines+markers+text",
                                            line=dict(color='firebrick', width=4),
                                            textposition="bottom center",
                                            textfont=dict(
                                                family="sans serif",
                                                size=30,
                                                color="crimson")),
                                      'layout': go.Layout(legend=dict(
                                                          orientation="h",
                                                          yanchor="bottom",
                                                          y=1.02,
                                                          xanchor="right",
                                                          x=1
                                                      ))}
                   for _f in ['daily_funnel', 'hourly_funnel', 'weekly_funnel', 'monthly_funnel',
                              'daily_funnel_downloads', 'hourly_funnel_downloads',
                              'weekly_funnel_downloads', 'monthly_funnel_downloads']},
        # not any recent KPIs for now
        "kpis": {}
    },
    "cohort": {
        # charts - Cohorts From 1, 2, 3 to 2, 3 ,4 Downloads to 1st Orders, daily, Weekly
        "charts": {_c: {'trace': go.Heatmap(z=[], x=[], y=[], colorscale='Viridis', opacity=0.9,
                                            showlegend=False, ygap=2, xgap=2, hoverongaps=None, showscale=False),
                        'annotation': go.Annotation(text=[], x=[], y=[], xref='x1', yref='y1', showarrow=False),
                        'layout': go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            width=1000, margin=dict(l=1, r=1, t=1, b=1), height=600,
                                            annotations=[])}
                   for _c in ['daily_cohort_downloads', 'daily_cohort_from_1_to_2',
                              'daily_cohort_from_2_to_3', 'daily_cohort_from_3_to_4',
                              'weekly_cohort_downloads', 'weekly_cohort_from_1_to_2',
                              'weekly_cohort_from_2_to_3', 'weekly_cohort_from_3_to_4']},
        # not any recent KPIs for now
        "kpis": {}
    },
    "stats": {
        # Descriptive Statistics
        "charts": {_f: {'trace': go.Scatter(mode="lines+markers+text",
                                            line=dict(color='firebrick', width=4),
                                            textposition="bottom center",
                                            textfont=dict(
                                                family="sans serif",
                                                size=30,
                                                color="crimson")),
                        'layout': go.Layout(legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ))}
                   for _f in ['daily_orders', 'weekly_orders', 'monthly_orders', 'hourly_orders']},
        # not any recent KPIs for now
        "kpis": {}
    },
    "descriptive": {
        # Descriptive Statistics
        "charts": {_f: {'trace': go.Scatter(mode="lines+markers+text",
                                            line=dict(color='firebrick', width=4),
                                            textfont=dict(
                                                family="sans serif",
                                                size=8,
                                                color="crimson")),
                        'layout': go.Layout(legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ))} if _f.split("_")[0] == 'weekly' else
                        {'trace': go.Bar(x=[], y=[]), 'layout': go.Layout()}
                   for _f in ["weekly_average_session_per_user",
                              "weekly_average_order_per_user", "purchase_amount_distribution",
                              "weekly_average_payment_amount"]},


        "kpis": {}
    },


}


class SampleData:
    """
    This enables us to visualize dashboards when there is no initial data has been created yet.
    These sample data comes from the sample data_folder default in the library.
    """
    kpis = {}
    folder = join(abspath_for_sample_data(), "exploratory_analysis", 'sample_data', '')
    for f in listdir(dirname(folder)):
        if f.split(".")[1] == 'csv':
            kpis["_".join(f.split(".")[0].split("_")[2:])] = None

    folder = join(abspath_for_sample_data(), "exploratory_analysis", 'sample_data', '')
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


def cohort_human_readable_form(cohort, tp):
    cohort_updated = pd.DataFrame()
    cohort_days = [int(i) for i in list(set(list(cohort.columns)) - set([tp]))]

    dates = list(zip(list(range(len(list(cohort[tp])))), reversed(list(cohort[tp]))))
    days_back = 15
    while len(cohort_updated) == 0:
        if days_back <= max(cohort_days):
            cohort_updated = cohort[[tp] + [str(i) for i in list(range(days_back+1))]]
            _days = list(map(lambda x: x[1], filter(lambda x: x[0] <= days_back, dates)))
            cohort_updated = cohort_updated[cohort_updated[tp].isin(_days)]
        days_back -= 1

    cohort_updated = cohort_updated.sort_values(by=tp, ascending=True)
    return cohort_updated


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
        if chart in ["_".join([t, 'orders']) for t in time_periods]:
            try:
                _t = 'date' if chart.split("_")[0] not in list(_data.columns) else chart.split("_")[0]
                _data = _data.sort_values(by=_t, ascending=True)
                trace['x'] = list(_data[_t])
                trace['y'] = list(_data['orders'])
                if _t not in ['daily', 'weekly']:
                    trace['text'] = list(_data['orders'])
            except Exception as e:
                print(e)
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
            trace = []
            for _a in set(list(_data.columns)) - set(time_periods):
                trace += [go.Scatter(x=list(_data[_tp]),
                                     y=list(_data[_a]),
                                     mode="lines+markers+text",
                                     name=_a,
                                     line=dict( width=4),
                                     textposition="bottom center",
                                     textfont=dict(
                                         family="sans serif",
                                         size=30))]
        if 'cohort' in chart.split("_"):
            _t = chart.split("_")[0]
            _t_str = ' day' if _t == 'daily' else ' week'
            _data = cohort_human_readable_form(_data, _t)
            z = array(_data[_data.columns[1:]]).tolist()
            x = [str(col) + _t_str for col in list(_data.columns)][1:]
            y = [str(ts)[0:10] for ts in list(_data[_data.columns[0]])]
            trace['z'], trace['x'], trace['y'] = z, x, y
        if chart in ["weekly_average_session_per_user", "weekly_average_order_per_user",
                     "purchase_amount_distribution", "weekly_average_payment_amount"]:
            if 'distribution' in chart.split("_"):
                _data['payment_bins'] = _data['payment_bins'].apply(lambda x: round(float(x), 2))
                _data['orders'] = _data['orders'].apply(lambda x: int(x))
                _trace_updated = []
                for _bin in _data.to_dict('results'):
                    _trace_updated.append(go.Bar(x=[_bin['payment_bins']], y=[_bin['orders']], showlegend=False))
                trace = _trace_updated
            else:
                _t = 'weekly'
                indicator = list(set(list(_data.columns)) - set([_t]))[0]
                _data = _data.sort_values(by=_t, ascending=True)
                trace['x'] = list(_data[_t])
                trace['y'] = list(_data[indicator])
                trace['text'] = list(_data[indicator])

        return [trace] if len(set(['funnel', 'distribution']) & set(chart.split("_"))) == 0 else trace

    def get_layout(self, layout, chart, annotation=None):
        _data = self.get_data(chart)
        if 'cohort' in chart.split("_"):
            _t = chart.split("_")[0]
            _t_str = ' day' if _t == 'daily' else ' week'
            _data = cohort_human_readable_form(_data, _t)
            z = array(_data[_data.columns[1:]]).tolist()
            x = [str(col) + _t_str for col in list(_data.columns)][1:]
            y = [str(ts)[0:10] for ts in list(_data[_data.columns[0]])]
            annotations = []
            for n, row in enumerate(z):
                for m, val in enumerate(row):
                    annotation['text'] = str(z[n][m])
                    try:
                        asd = x[m]
                        annotation['x'] = x[m]
                    except Exception as e:
                        print(e)
                    annotation['y'] = y[n]
                    annotations.append(annotation)
            layout['annotations'] = annotations
        return [layout]

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
            layout = self.get_layout(charts[target]['charts'][c]['layout'], c,
                                     annotation=charts[target]['charts'][c]['annotation'] if target == 'cohort' else None)
            self.graph_json['charts'][c] = {'trace': trace,
                                            'layout': layout}
        # collecting KPIs
        self.graph_json['kpis'] = {}
        for k in charts[target]['kpis']:
            _obj = self.get_values(k)
            for _k in charts[target]['kpis'][k]:
                self.graph_json['kpis'][_k] = '{:,}'.format(int(_obj[_k])).replace(",", ".")
        return self.graph_json

    def get_json_format(self, chart):
        return json.dumps({"trace": chart['trace'],
                           "layout": chart['layout']}, cls=plotly.utils.PlotlyJSONEncoder)