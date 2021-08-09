import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from web.app.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
import json

from customeranalytics.web.app.home.models import RouterRequest
from customeranalytics.web.app.home.search import Search
from customeranalytics.web.app.home.forms import SampleData, RealData, Charts, charts
from customeranalytics.web.app.home.profiles import Profiles
from customeranalytics.data_storage_configurations.logger import LogsBasicConfeger


LogsBasicConfeger()
samples = SampleData()
real = RealData()
charts = Charts(samples.kpis, real)
reqs = RouterRequest()
profile = Profiles()
search = Search()


def get_segment(request):
    try:
        segment = request.path.split('/')[-1]
        if segment == '':
            segment = 'index'
        return segment
    except:
        return None


@blueprint.route('/search', methods=["GET", "POST"])
@login_required
def search_data():
    """
    When logged In, Platform start with General Dashboard running on index.html
    :return: render_template
    """
    pic = profile.fetch_pic()
    search_value = dict(request.form).get('search', '')
    print("search")
    results = search.search_results(search_value)
    # print(search_type)
    graph_json, data_type, filters = charts.get_chart(target='search_' + results['search_type'])
    chart_names = search.get_search_chart_names(results['search_type'])
    kpis = search.convert_kpi_names_to_numeric_names(graph_json)
    search.delete_search_data(results)
    return render_template('search.html',
                           segment='search',
                           pic=pic,
                           chart_2=charts.get_json_format(graph_json['charts']['chart_2_search']),
                           chart_3=charts.get_json_format(graph_json['charts']['chart_3_search']),
                           chart_4=charts.get_json_format(graph_json['charts']['chart_4_search']),
                           kpis=kpis,
                           chart_names=chart_names,
                           search_results=results,
                           data_type=data_type)


@blueprint.route('/index', methods=["GET", "POST"])
@login_required
def index():
    """
    When logged In, Platform start with General Dashboard running on index.html
    :return: render_template
    """
    pic = profile.fetch_pic()
    index = dict(request.form).get('index', 'main')
    date = dict(request.form).get('date', None)
    graph_json, data_type, filters = charts.get_chart(target='index', index=index, date=date)  # collect charts on index.html
    return render_template('index.html',
                           segment='index',
                           pic=pic,
                           charts=charts.get_json_format(graph_json['charts']['daily_orders']),
                           customer_segments=charts.get_json_format(graph_json['charts']['segmentation']),
                           customer_journey=charts.get_json_format(graph_json['charts']['customer_journey']),
                           top_products=charts.get_json_format(graph_json['charts']['most_ordered_products']),
                           top_categories=charts.get_json_format(graph_json['charts']['most_ordered_categories']),
                           churn=charts.get_json_format(graph_json['charts']['churn']),
                           churn_weekly=charts.get_json_format(graph_json['charts']['churn_weekly']),
                           kpis=graph_json['kpis'],
                           data_type=data_type,
                           filters=filters)


@blueprint.route("/upload-image", methods=["POST"])
@login_required
def upload_image():
    segment = get_segment(request)
    if request.method == "POST":
        profile.add_pic(request)
    args = profile.fetch_chats()
    pic = profile.fetch_pic()
    try:
        return render_template("profile.html",
                        segment=segment,
                        pic=pic, messages=args['messages'], chart=args['charts'], filters=args['filters'])
    except TemplateNotFound:
        return render_template('page-404.html'), 404
    except Exception as e:
        return render_template('page-500.html'), 500


@blueprint.route('/<template>', methods=['GET', 'POST'])
@login_required
def route_template(template):
    """
    page router;
        This will keep updated... (Work in proggress)


    :param template: .../index, ../manage-data
    :return: render_template
    """
    pic = profile.fetch_pic()
    try:
        if not template.endswith( '.html' ):
            template += '.html'

        segment = get_segment(request)

        index = dict(request.form).get('index', 'main')
        date = dict(request.form).get('date', None)

        if template in ['funnel-session.html', 'funnel-customer.html']:
            additional_name = '' if template == 'funnel-session.html' else '_downloads'
            graph_json, data_type, filters = charts.get_chart(target='funnel', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   daily_funnel=charts.get_json_format(
                                       graph_json['charts']['daily_funnel' + additional_name]),
                                   weekly_funnel=charts.get_json_format(
                                       graph_json['charts']['weekly_funnel' + additional_name]),
                                   monthly_funnel=charts.get_json_format(
                                       graph_json['charts']['monthly_funnel' + additional_name]),
                                   hourly_funnel=charts.get_json_format(
                                       graph_json['charts']['hourly_funnel' + additional_name]),
                                   data_type=data_type,
                                   filters=filters)
        if template == 'cohorts.html':
            graph_json, data_type, filters = charts.get_chart(target='cohort', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
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
                                       graph_json['charts']['weekly_cohort_from_3_to_4']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'stats-purchase.html':
            graph_json, data_type, filters = charts.get_chart(target='stats', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   daily_orders=charts.get_json_format(graph_json['charts']['daily_orders']),
                                   weekly_orders=charts.get_json_format(graph_json['charts']['weekly_orders']),
                                   monthly_orders=charts.get_json_format(graph_json['charts']['monthly_orders']),
                                   hourly_orders=charts.get_json_format(graph_json['charts']['hourly_orders']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'stats-desc.html':
            graph_json, data_type, filters = charts.get_chart(target='descriptive', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   weekly_average_session_per_user=charts.get_json_format(
                                       graph_json['charts']['weekly_average_session_per_user']),
                                   weekly_average_order_per_user=charts.get_json_format(
                                       graph_json['charts']['weekly_average_order_per_user']),
                                   purchase_amount_distribution=charts.get_json_format(
                                       graph_json['charts']['purchase_amount_distribution']),
                                   weekly_average_payment_amount=charts.get_json_format(
                                       graph_json['charts']['weekly_average_payment_amount']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'abtest-promotion.html':
            graph_json, data_type, filters = charts.get_chart(target='abtest-promotion', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic="avatar-2.png",
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
                                   data_type=data_type,
                                   filters=filters)

        if template == 'abtest-product.html':
            graph_json, data_type, filters = charts.get_chart(target='abtest-product', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   product_use_ba_a_accept=charts.get_json_format(
                                       graph_json['charts']['product_usage_before_after_amount_accept']),
                                   product_use_ba_a_reject=charts.get_json_format(
                                       graph_json['charts']['product_usage_before_after_amount_reject']),
                                   product_use_ba_o_accept=charts.get_json_format(
                                       graph_json['charts']['product_usage_before_after_orders_accept']),
                                   product_use_ba_o_reject=charts.get_json_format(
                                       graph_json['charts']['product_usage_before_after_orders_reject']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'abtest-segments.html':
            graph_json, data_type, filters = charts.get_chart(target='abtest-segments', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   sc_weekly_ba_orders=charts.get_json_format(
                                       graph_json['charts']['segments_change_weekly_before_after_orders']),
                                   sc_daily_ba_orders=charts.get_json_format(
                                       graph_json['charts']['segments_change_daily_before_after_orders']),
                                   sc_monthly_ba_orders=charts.get_json_format(
                                       graph_json['charts']['segments_change_monthly_before_after_orders']),
                                   sc_weekly_ba_amount=charts.get_json_format(
                                       graph_json['charts']['segments_change_weekly_before_after_amount']),
                                   sc_daily_ba_amount=charts.get_json_format(
                                       graph_json['charts']['segments_change_daily_before_after_amount']),
                                   sc_monthly_ba_amount=charts.get_json_format(
                                       graph_json['charts']['segments_change_monthly_before_after_amount']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'product.html':
            graph_json, data_type, filters = charts.get_chart(target='product_analytic', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   most_combined_products=charts.get_json_format(
                                       graph_json['charts']['most_combined_products']),
                                   most_ordered_products=charts.get_json_format(
                                       graph_json['charts']['most_ordered_products']),
                                   most_ordered_categories=charts.get_json_format(
                                       graph_json['charts']['most_ordered_categories']),
                                   data_type=data_type,
                                   filters=filters)
        if template == 'rfm.html':
            graph_json, data_type, filters = charts.get_chart(target='rfm', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   rfm=charts.get_json_format(graph_json['charts']['rfm']),
                                   frequency_recency=charts.get_json_format(graph_json['charts']['frequency_recency']),
                                   monetary_frequency=charts.get_json_format(graph_json['charts']['monetary_frequency']),
                                   recency_monetary=charts.get_json_format(graph_json['charts']['recency_monetary']),
                                   data_type=data_type,
                                   filters=filters
                                   )

        if template == 'customer-segmentation.html':
            graph_json, data_type, filters = charts.get_chart(target='customer-segmentation', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   segmentation=charts.get_json_format(graph_json['charts']['segmentation']),
                                   frequency_clusters=charts.get_json_format(graph_json['charts']['frequency_clusters']),
                                   monetary_clusters=charts.get_json_format(graph_json['charts']['monetary_clusters']),
                                   recency_clusters=charts.get_json_format(graph_json['charts']['recency_clusters']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'index2.html':
            graph_json, data_type, filters = charts.get_chart(target='index2', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   rfm=charts.get_json_format(
                                       graph_json['charts']['rfm']),
                                   purchase_amount_distribution=charts.get_json_format(
                                       graph_json['charts']['purchase_amount_distribution']),
                                   uoc_order_seq=charts.get_json_format(
                                       graph_json['charts']['user_counts_per_order_seq']),
                                   daily_funnel=charts.get_json_format(
                                       graph_json['charts']['daily_funnel']),
                                   weekly_cohort_downloads=charts.get_json_format(
                                       graph_json['charts']['weekly_cohort_downloads']),
                                   daily_clv=charts.get_json_format(
                                       graph_json['charts']['daily_clv']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'clv.html':
            graph_json, data_type, filters = charts.get_chart(target='clv', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   daily_clv=charts.get_json_format(
                                       graph_json['charts']['daily_clv']),
                                   clvsegments_amount=charts.get_json_format(
                                       graph_json['charts']['clvsegments_amount']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'anomaly.html':
            graph_json, data_type, filters = charts.get_chart(target='anomaly', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   pic=pic,
                                   dfunnel_anomaly=charts.get_json_format(
                                       graph_json['charts']['dfunnel_anomaly']),
                                   dcohort_anomaly=charts.get_json_format(
                                       graph_json['charts']['dcohort_anomaly']),
                                   dcohort_anomaly_2=charts.get_json_format(
                                       graph_json['charts']['dcohort_anomaly_2']),
                                   dorders_anomaly=charts.get_json_format(
                                       graph_json['charts']['dorders_anomaly']),
                                   clvrfm_anomaly=charts.get_json_format(
                                       graph_json['charts']['clvrfm_anomaly']),
                                   data_type=data_type,
                                   filters=filters)

        if template not in ['funnel-customer.html', 'funnel-customer.html', 'index.html', 'index2.html', 'rfm.htm',
                            'product.html', 'abtest-segments.html', 'abtest-product.html', 'abtest-promotion.html',
                            'stats-desc.html', 'stats-purchase.htm', 'cohorts.html', 'customer-segmentation.html']:
            if template in ['profile.html', 'settings.html', 'upload.php.html']:
                profile.add_new_message(dict(request.form))
                args = profile.fetch_chats()
                return render_template(template,
                                       segment=segment,
                                       pic=pic, messages=args['messages'], chart=args['charts'], filters=args['filters'])
            else:
                reqs.execute_request(req=dict(request.form), template=segment)
                reqs.fetch_results(segment, dict(request.form))
                values = reqs.message
                return render_template(template, segment=segment, pic=pic, values=values)

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    except Exception as e:
        print(e)
        return render_template('page-500.html'), 500



