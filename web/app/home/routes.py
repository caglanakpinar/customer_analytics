import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from web.app.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
import json

from web.app.home.models import RouterRequest
from web.app.home.forms import SampleData, RealData, Charts, charts
from screeninfo import get_monitors


samples = SampleData()
real = RealData()
real.execute_real_data()
charts = Charts(samples.kpis, real.kpis)
reqs = RouterRequest()


def get_segment(request):
    try:
        segment = request.path.split('/')[-1]
        if segment == '':
            segment = 'index'
        return segment
    except:
        return None


@blueprint.route('/index', methods=["GET", "POST"])
@login_required
def index():
    """
    When logged In, Platform start with General Dashboard running on index.html
    :return: render_template
    """
    graph_json = charts.get_chart(target='index') # collect charts on index.html
    return render_template('index.html',
                           segment='index',
                           charts=charts.get_json_format(graph_json['charts']['daily_orders']),
                           customer_segments=charts.get_json_format(graph_json['charts']['customer_segmentation']),
                           kpis=graph_json['kpis']
                           )


@blueprint.route('/<template>', methods=['GET', 'POST'])
@login_required
def route_template(template):
    """
    page router;
        This will keep updated... (Work in proggress)


    :param template: .../index, ../manage-data
    :return: render_template
    """

    try:
        if not template.endswith( '.html' ):
            template += '.html'

        segment = get_segment(request)

        if template in ['funnel-session.html', 'funnel-customer.html']:
            additional_name = '' if template == 'funnel-session.html' else '_downloads'
            graph_json = charts.get_chart(target='funnel')
            return render_template(template,
                                   segment=segment,
                                   daily_funnel=charts.get_json_format(
                                       graph_json['charts']['daily_funnel' + additional_name]),
                                   weekly_funnel=charts.get_json_format(
                                       graph_json['charts']['weekly_funnel' + additional_name]),
                                   monthly_funnel=charts.get_json_format(
                                       graph_json['charts']['monthly_funnel' + additional_name]),
                                   hourly_funnel=charts.get_json_format(
                                       graph_json['charts']['hourly_funnel' + additional_name])
                                   )
        if template == 'cohorts.html':
            graph_json = charts.get_chart(target='cohort')
            asd = charts.get_json_format(
                                       graph_json['charts']['daily_cohort_downloads'])
            return render_template(template,
                                   segment=segment,
                                   daily_cohort_downloads=charts.get_json_format(
                                       graph_json['charts']['daily_cohort_downloads']),
                                   daily_cohort_from_1_to_2=charts.get_json_format(
                                       graph_json['charts']['daily_cohort_from_1_to_2']),
                                   daily_cohort_from_2_to_3=charts.get_json_format(
                                       graph_json['charts']['daily_cohort_from_2_to_3']),
                                   daily_cohort_from_3_to_4=charts.get_json_format(
                                       graph_json['charts']['daily_cohort_from_3_to_4']),
                                   weekly_cohort_downloads=charts.get_json_format(
                                       graph_json['charts']['weekly_cohort_downloads']),
                                   weekly_cohort_from_1_to_2=charts.get_json_format(
                                       graph_json['charts']['weekly_cohort_from_1_to_2']),
                                   weekly_cohort_from_2_to_3=charts.get_json_format(
                                       graph_json['charts']['weekly_cohort_from_2_to_3']),
                                   weekly_cohort_from_3_to_4=charts.get_json_format(
                                       graph_json['charts']['weekly_cohort_from_3_to_4'])
                                   )

        if template == 'stats-purchase.html':
            graph_json = charts.get_chart(target='stats')
            return render_template(template,
                                   segment=segment,
                                   daily_orders=charts.get_json_format(graph_json['charts']['daily_orders']),
                                   weekly_orders=charts.get_json_format(graph_json['charts']['weekly_orders']),
                                   monthly_orders=charts.get_json_format(graph_json['charts']['monthly_orders']),
                                   hourly_orders=charts.get_json_format(graph_json['charts']['hourly_orders'])
                                   )

        if template == 'stats-desc.html':
            graph_json = charts.get_chart(target='descriptive')
            return render_template(template,
                                   segment=segment,
                                   weekly_average_session_per_user=charts.get_json_format(graph_json['charts']['weekly_average_session_per_user']),
                                   weekly_average_order_per_user=charts.get_json_format(graph_json['charts']['weekly_average_order_per_user']),
                                   purchase_amount_distribution=charts.get_json_format(graph_json['charts']['purchase_amount_distribution']),
                                   weekly_average_payment_amount=charts.get_json_format(graph_json['charts']['weekly_average_payment_amount'])
                                   )

        if template in ['abtest-products.html', 'abtest-promotion.html', 'abtest-segments.html']:
            graph_json = charts.get_chart(target='abtest-promotion')
            return render_template(template,
                                   segment=segment,
                                   o_pa_diff=charts.get_json_format(
                                       graph_json['charts']['order_and_payment_amount_differences']),
                                   promotion_comparison=charts.get_json_format(
                                       graph_json['charts']['promotion_comparison']),
                                   promo_use_ba_a_accept=charts.get_json_format(
                                       graph_json['charts']['promotion_usage_before_after_amount_accept']),
                                   promo_use_ba_a_reject=charts.get_json_format(
                                       graph_json['charts']['promotion_usage_before_after_amount_reject']),
                                   promo_use_ba_o_accept=charts.get_json_format(
                                       graph_json['charts']['promotion_usage_before_after_orders_accept']),
                                   promo_use_ba_o_reject=charts.get_json_format(
                                       graph_json['charts']['promotion_usage_before_after_orders_reject']),
                                   )

        if template == 'index2.html':
            graph_json = charts.get_chart(target='index2')
            return render_template(template, segment=segment, rfm=charts.get_json_format(graph_json['charts']['rfm']))
        if template not in ['funnel-customer.html', 'funnel-customer.html', 'index.html', 'index2.html']:
            reqs.execute_request(req=dict(request.form), template=segment)
            reqs.fetch_results(segment, dict(request.form))
            values = reqs.message
            return render_template(template, segment=segment, values=values)

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except Exception as e:
        print(e)
        return render_template('page-500.html'), 500



