import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pandas as pd
from numpy import array
import json

from os.path import join, dirname, exists
from os import listdir
import plotly.graph_objs as go
import plotly
from screeninfo import get_monitors
from sqlalchemy import create_engine, MetaData

from utils import convert_to_day, abspath_for_sample_data
from configs import time_periods, descriptive_stats, abtest_promotions, abtest_products

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
        # charts - orders_monthly, orders_hourly, orders_weekly, orders_daily, segmentation
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
             'segmentation': {'trace': go.Treemap(
                 marker_colorscale='Blues',
                 textinfo="label+value+percent parent+percent entry", parents=[""] * 7),
                 'layout': go.Layout(width=1000,
                                     margin=dict(l=1, r=1, t=1, b=1),
                                     height=400)},
             "customer_journey": {'trace': go.Scatter(x=[], y=[],
                                                      marker=dict(size=[], color=[]), mode='markers', name='markers'),
                                  'layout': go.Layout()},
             "most_ordered_products": {'trace': go.Bar(x=[], y=[]), 'layout': go.Layout()},
             "most_ordered_categories": {'trace': go.Bar(x=[], y=[]), 'layout': go.Layout()},
             },
        ## TODO  organic - promoted ratio
        # KPIs - total_orders, total_visits, total_unique_visitors. index.html left top
        "kpis": {"kpis": ['total_orders', 'total_visitors', 'total_revenue', 'total_discount']}},
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
                           },
                   "user_counts_per_order_seq": {'trace': go.Bar(),
                                                  'layout': go.Layout(
                                                      xaxis_title="X Axis Title",
                                                      yaxis_title="Y Axis Title",
                                                      margin=dict(r=1, t=1))}



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
    "abtest-promotion": {
        # Descriptive Statistics
        "charts": {_f: {'trace': go.Bar(),
                        'layout': go.Layout(
                            legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1),
                            margin=dict(r=1, t=1))} if _f not in ["order_and_payment_amount_differences",
                                                                       "promotion_comparison"] else
                       {'trace': go.Scatter(x=[], y=[], marker=dict(size=[], color=[]), mode='markers', name='markers'),
                        'layout': go.Layout(margin=dict(l=1, r=1, t=1, b=1))}
                   for _f in ["order_and_payment_amount_differences", "promotion_comparison",
                              "promotion_usage_before_after_amount_accept", "promotion_usage_before_after_amount_reject",
                              "promotion_usage_before_after_orders_accept", "promotion_usage_before_after_orders_reject"]},
        "kpis": {}
    },
    "abtest-product": {
        # Descriptive Statistics
        "charts": {_f: {'trace': go.Bar(),
                        'layout': go.Layout(
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1),
                            margin=dict(r=1, t=1))} if _f not in ["order_and_payment_amount_differences",
                                                                  "promotion_comparison"] else
        {'trace': go.Scatter(x=[], y=[], marker=dict(size=[], color=[]), mode='markers', name='markers'),
         'layout': go.Layout(margin=dict(l=1, r=1, t=1, b=1))}
                   for _f in ["product_usage_before_after_amount_accept",
                              "product_usage_before_after_amount_reject",
                              "product_usage_before_after_orders_accept",
                              "product_usage_before_after_orders_reject"]},
        "kpis": {}
    },
    "abtest-segments": {
        # Descriptive Statistics
        "charts": {_f: {'trace': [], 'layout': []} for _f in
                   ['segments_change_weekly_before_after_orders', 'segments_change_weekly_before_after_amount',
                    'segments_change_daily_before_after_orders', 'segments_change_daily_before_after_amount',
                    'segments_change_monthly_before_after_orders', 'segments_change_monthly_before_after_amount']},
        "kpis": {}
    },
    "product_analytic": {
        # product analytics
        "charts": {
            "most_combined_products": {'trace': go.Bar(x=[], y=[]), 'layout': go.Layout()},
            "most_ordered_products": {'trace': go.Bar(x=[], y=[]), 'layout': go.Layout()},
            "most_ordered_categories": {'trace': go.Bar(x=[], y=[]), 'layout': go.Layout()},
        },
        # not any recent KPIs for now
        "kpis": {}
    },
    "rfm": {
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
                           },
                   'frequency_recency': {'trace': go.Scatter(y=[], x=[],
                                                    mode='markers',
                                                    marker=dict(
                                                        size=16,
                                                        color=[],
                                                        colorscale='Viridis',  # one of plotly colorscales
                                                        showscale=False
                                                    )),
                                         "layout": go.Layout(margin=dict(l=1, r=1, t=1, b=1))},
                   'monetary_frequency': {'trace': go.Scatter(y=[], x=[],
                                                    mode='markers',
                                                    marker=dict(
                                                        size=16,
                                                        color=[],
                                                        colorscale='Viridis',  # one of plotly colorscales
                                                        showscale=False
                                                    )),
                                          "layout": go.Layout(margin=dict(l=1, r=1, t=1, b=1))},
                   'recency_monetary': {'trace': go.Scatter(y=[], x=[],
                                                    mode='markers',
                                                    marker=dict(
                                                        size=16,
                                                        color=[],
                                                        colorscale='Viridis',  # one of plotly colorscales
                                                        showscale=False
                                                    )),
                                        "layout": go.Layout(margin=dict(l=1, r=1, t=1, b=1))}

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
    After the scheduling process is done,
    reports are created on the temporary folder with a folder name 'build_in_reports'.
    This collects all reports of  created .csv files from 'build_in_reports'.
    Each dimension and whole data ('main' folder name) will be stored separately.
    Each dimension of reports will be created as .csv file.
    """
    kpis = {}
    es_tag = pd.read_sql("SELECT * FROM es_connection", con).to_dict('resutls')[-1]
    folder = join(es_tag['directory'], "build_in_reports", "")
    
    def check_for_the_report(self, report_name, index='main'):
        """
        checks for 'build_in_reports' while platform is running.
        """
        try:
            es_tag = pd.read_sql("SELECT * FROM es_connection", con).to_dict('resutls')[-1]
            return exists(join(es_tag['directory'], "build_in_reports", index, report_name + ".csv"))
        except:
            return False

    def fetch_report(self, report_name, index='main'):
        """
        checks for 'build_in_reports' while platform is running and collect the selected report.
        """
        try:
            es_tag = pd.read_sql("SELECT * FROM es_connection", con).to_dict('resutls')[-1]
            return pd.read_csv(join(es_tag['directory'], "build_in_reports", index, report_name + ".csv"))
        except:
            return False

    # this will collect the report in the 'build_in_reports'.
    # whole data of reports will be stored in 'main' folder. dimensions are stored seperatelly
    try:
        for index in listdir(dirname(folder)):
            _folder = join(es_tag['directory'], "build_in_reports", index, "")
            kpis[index] = {}
            for f in listdir(dirname(_folder)):
                try:
                    kpis[index][f.split(".")[0]] = pd.read_csv(join(_folder, f))
                except Exception as e:
                    print(e)
    except Exception as e:
        print(e)
            

def cohort_human_readable_form(cohort, tp):
    """
    cohorts data is manipulated in or order to convert more readable format.
    """
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
    def __init__(self, samples, real):
        """
        The main perspective here to store each chart serialized .json file into the self.graph_json
        e.g. index.html has segmentation chart
            self.graph_json['charts']['segmentation'] = {'trace': [], 'layout': {}}
        If each chart needs any specific change pls check **charts** dictionary.
        :param samples: built-in data sets in .csv format, converted to pandas data-frame
        :param reals: created ports in .csv format, converted to pandas data-frame
        """
        self.samples = samples
        self.reals = real
        self.graph_json = {}
        self.monitor = get_monitors()[0]
        self.descriptive_stats = descriptive_stats
        self.abtest_promotions = abtest_promotions
        self.abtest_products = abtest_products

    def get_data(self, chart, index):
        """
        checks both sample and real data. If there is real data for the related KPI or chart fetches from sample_data.
        :param chart: e.g. rfm, segmentation, ...
        :return:
        """
        try:
            if chart not in list(self.reals.kpis.keys()):
                if not self.reals.check_for_the_report(report_name=chart, index=index):
                    return self.samples[chart], False
                else:
                    return self.reals.fetch_report(report_name=chart, index=index), True
            else:
                return self.reals.kpis[index][chart], True
        except Exception as e:
            print()
            return self.samples[chart], False

    def get_widths_heights(self, target, chart):
        if chart == 'segmentation':
            if self.monitor.height == 1080 and self.monitor.width == 1920:
                width, height = 720, 450
            if self.monitor.height == 1050 and self.monitor.width == 1680:
                width, height = 1000, 400
        if chart == 'rfm':
            if self.monitor.height == 1080 and self.monitor.width == 1920:
                width, height = 700, 600
            if self.monitor.height == 1050 and self.monitor.width == 1680:
                width, height = 1000, 600
        if chart in ['segmentation', 'rfm']:
            charts[target]['charts'][chart]['layout']['width'] = width
            charts[target]['charts'][chart]['layout']['height'] = height

    def decide_trace_type(self, trace, chart):
        if len(set(['funnel', 'distribution']) & set(chart.split("_"))) != 0:
            return trace
        else:
            if type(trace) == list:
                return trace
            else:
                return [trace]

    def ab_test_of_trace(self, data, chart):
        """
        "order_and_payment_amount_differences",
        "promotion_comparison",
        "promotion / product  _usage_before_after_amount_accept",
        "promotion / product  _usage_before_after_amount_reject",
        "promotion / product  _usage_before_after_orders_accept",
        "promotion / product  _usage_before_after_orders_reject"
        """
        _trace = []
        if chart.split("_")[1] == 'usage':
            _name = 'order count' if chart.split("_")[-2] == 'orders' else 'purchase amount'
            names = ["before average "+_name+"  per c.", "after average "+_name+" per c."]
            indicator = chart.split("_")[0] + 's'
            _trace = [
                go.Bar(name=names[0], x=data[indicator], y=data['mean_control']),
                go.Bar(name=names[1], x=data[indicator], y=data['mean_validation'])
                ]
        if chart == 'order_and_payment_amount_differences':
            data = data.rename(columns={"diff": "Difference of Order (Before Vs After)",
                                        "diff_amount": "Difference of Payment Amount (Before Vs After)"})
            _trace = go.Scatter(x=data['Difference of Order (Before Vs After)'],
                                y=data['Difference of Payment Amount (Before Vs After)'], mode='markers',
                                marker=dict(color=list(range(len(data))), colorscale='Rainbow'))
        if chart == 'promotion_comparison':
            _trace = go.Scatter(x=data['accept_Ratio'],
                                y=data['total_effects'],
                                text=data['1st promo'],
                                marker=dict(size=data['total_negative_effects'],
                                            color=list(range(len(data))), colorscale='Rainbow'),
                                mode='markers',
                                name='markers')
        if chart.split("_")[1] == 'change':
            _trace = data.to_dict('results')
        return _trace

    def get_trace(self, trace, chart, index):
        """
        fill more variables on charts dictionary. At this process, data sets are stored in the trace.

        :param trace: e.g. [{go.Scatter()}]
        :param chart: e.g. rfm, segmentation, ...
        :return:
        """
        _data, is_real_data = self.get_data(chart, index)  # collect data
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
        if chart == 'segmentation':
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
        if chart in self.descriptive_stats:
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
        if chart in self.abtest_promotions + self.abtest_products:
            trace = self.ab_test_of_trace(_data, chart)
        if chart == 'user_counts_per_order_seq':
            trace['x'] = list(_data['order_seq_num'])
            trace['y'] = list(_data['frequency'])
        if chart == 'customer_journey':
            _data = _data.reset_index().iloc[:-1]
            _data['text'] = _data.apply(
                lambda row: 'Customers Who have ' + str(int(row['index'])) + ' orders. Avg. Purchase Amount : ' + str(round(row['customers` average Purchase Value'], 2)) + "  || Avg. Duration between last and recent order :" + str(round(row['hourly order differences'], 2)), axis=1)
            trace = go.Scatter(x=_data['index'],
                               y=_data['hourly order differences'],
                               text=_data['text'],
                               marker=dict(size=_data['customers` average Purchase Value'],
                                           color=list(range(len(_data))), colorscale='Rainbow'),
                               mode='markers',
                               name='markers')
        if 'most' in chart.split("_"):
            x_column = 'products' if 'products' in chart.split("_") else 'category'
            trace['x'], trace['y'] = list(_data[x_column]), list(_data['order_count'])
        if 'recency' in chart.split("_") or 'frequency' in chart.split("_") or 'monetary' in chart.split("_"):
            trace['x'] = list(_data[chart.split("_")[0]])
            trace['y'] = list(_data[chart.split("_")[1]])
            trace['marker']['color'] = list(_data['segments_numeric'])  # segments are numerical values.

        return self.decide_trace_type(chart=chart, trace=trace), is_real_data

    def get_layout(self, layout, chart, index, annotation=None):
        _data = self.get_data(chart, index)[0]
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
                        annotation['x'] = x[m]
                    except Exception as e:
                        print(e)
                    annotation['y'] = y[n]
                    annotations.append(annotation)
            layout['annotations'] = annotations

        if 'usage' in chart.split("_") or chart == 'customer_journey':
            columns = ['mean_control', 'mean_validation']
            columns = ['hourly order differences'] * 2 if chart == 'customer_journey' else columns
            layout['yaxis'] = {"range": [
                round(max(0, min(min(_data[columns[0]]), min(_data[columns[1]])) - 0.02), 2),
                round(max(0, max(max(_data[columns[0]]), max(_data[columns[1]])) - 0.02), 2)]}

        return [layout] if 'usage' not in chart.split("_") else layout

    def get_values(self, kpi, index):
        """
        get data related KPI
        :param kpi: .e.g.total_orders, total_visitors, ...
        :return: dictionary with KPIs in keys
        """
        _data = self.get_data(kpi, index)[0].to_dict('results')[0]
        return _data

    def get_chart(self, target, index='main'):
        """
        related to 'target', charts and KPIs are collected in order to show on-page.
        The main aim here, fill the self.graph_json dictionary with serialized dictionaries.
        :param target:
        :return:
        """
        # collecting charts
        self.graph_json['charts'] = {}
        for c in charts[target]['charts']:
            trace, is_real_data = self.get_trace(charts[target]['charts'][c]['trace'], c, index)
            self.get_widths_heights(chart=c, target=target)
            layout = self.get_layout(charts[target]['charts'][c]['layout'], c,
                                     annotation=charts[target]['charts'][c]['annotation'] if target == 'cohort' else None, index=index)
            self.graph_json['charts'][c] = {'trace': trace,
                                            'layout': layout, 'is_real_data': is_real_data}
        # collecting KPIs
        self.graph_json['kpis'] = {}
        for k in charts[target]['kpis']:
            _obj = self.get_values(k, index)
            for _k in charts[target]['kpis'][k]:
                try:
                    self.graph_json['kpis'][_k] = '{:,}'.format(int(_obj[_k])).replace(",", ".")
                except Exception as e:
                    print()
        return self.graph_json

    def get_json_format(self, chart):
        return json.dumps({"trace": chart['trace'],
                           "layout": chart['layout']}, cls=plotly.utils.PlotlyJSONEncoder)