from customeranalytics.app.home import blueprint
from flask import render_template, request
from jinja2 import TemplateNotFound


from customeranalytics.app.home.models import RouterRequest
from customeranalytics.data_storage_configurations.logger import LogsBasicConfeger


LogsBasicConfeger()
router = RouterRequest()


def get_segment(request):
    try:
        segment = request.path.split('/')[-1]
        if segment == '':
            segment = 'index'
        return segment
    except:
        return None


@blueprint.route('/search', methods=["GET", "POST"])
def search_data():
    """
    This is for search result pages rendering with search.html
    :return: render_template
    """
    search_value = dict(request.form).get('search', '')
    results = router.search_results(search_value)
    graph_json, data_type, filters = router.get_chart(target='search_' + results['search_type'])
    chart_names = router.get_search_chart_names(results['search_type'])
    kpis = router.convert_kpi_names_to_numeric_names(graph_json)
    router.update_message()
    router.delete_search_data(results)
    return render_template('search.html',
                           segment='search',
                           chart_2=router.get_json_format(graph_json['charts']['chart_2_search']),
                           chart_3=router.get_json_format(graph_json['charts']['chart_3_search']),
                           chart_4=router.get_json_format(graph_json['charts']['chart_4_search']),
                           kpis=kpis,
                           chart_names=chart_names,
                           search_results=results,
                           data_type=data_type)


@blueprint.route('/index', methods=["GET", "POST"])
def index():
    """
    When logged In, Platform start with General Dashboard running on index.html
    :return: render_template
    """
    index = dict(request.form).get('index', 'main')
    date = dict(request.form).get('date', None)
    graph_json, data_type, filters = router.get_chart(target='index', index=index, date=date)  # collect charts on index.html
    router.update_message()
    return render_template(
        'index.html',
        segment='index',
        charts=router.get_json_format(graph_json['charts']['daily_orders']),
        customer_segments=router.get_json_format(graph_json['charts']['segmentation']),
        customer_journey=router.get_json_format(graph_json['charts']['customer_journey']),
        top_products=router.get_json_format(graph_json['charts']['most_ordered_products']),
        top_categories=router.get_json_format(graph_json['charts']['most_ordered_categories']),
        churn=router.get_json_format(graph_json['charts']['churn']),
        churn_weekly=router.get_json_format(graph_json['charts']['churn_weekly']),
        kpis=graph_json['kpis'],
        data_type=data_type,
        filters=filters
    )

@blueprint.route('/<template>', methods=['GET', 'POST'])
def route_template(template):
    """
    page router;
        This will keep updated... (Work in proggress)


    :param template: .../index, ../manage-data
    :return: render_template
    """
    router.update_message()
    try:
        if not template.endswith( '.html' ):
            template += '.html'

        segment = get_segment(request)
        index = dict(request.form).get('index', 'main')
        date = dict(request.form).get('date', None)

        if template in ['funnel-session.html', 'funnel-customer.html']:
            additional_name = '' if template == 'funnel-session.html' else '_downloads'
            router.update_message()
            graph_json, data_type, filters = router.get_chart(target='funnel', index=index, date=date)
            return render_template(
                template,
                segment=segment,
                daily_funnel=router.get_json_format(
                    graph_json['charts']['daily_funnel' + additional_name]),
                weekly_funnel=router.get_json_format(
                    graph_json['charts']['weekly_funnel' + additional_name]),
                monthly_funnel=router.get_json_format(
                    graph_json['charts']['monthly_funnel' + additional_name]),
                hourly_funnel=router.get_json_format(
                    graph_json['charts']['hourly_funnel' + additional_name]),
                data_type=data_type,
                filters=filters
            )
        if template == 'cohorts.html':
            (
                graph_json,
                data_type,
                filters
            ) = router.get_chart(
                target='cohort',
                index=index,
                date=date
            )
            return render_template(
                template,
                segment=segment,
                daily_cohort_downloads=router.get_json_format(
                    graph_json['charts']['daily_cohort_downloads']),
                daily_cohort_from_1_to_2=router.get_json_format(
                    graph_json['charts']['daily_cohort_from_1_to_2']),
                daily_cohort_from_2_to_3=router.get_json_format(
                    graph_json['charts']['daily_cohort_from_2_to_3']),
                daily_cohort_from_3_to_4=router.get_json_format(
                    graph_json['charts']['daily_cohort_from_3_to_4']),
                weekly_cohort_downloads=router.get_json_format(
                    graph_json['charts']['weekly_cohort_downloads']),
                weekly_cohort_from_1_to_2=router.get_json_format(
                    graph_json['charts']['weekly_cohort_from_1_to_2']),
                weekly_cohort_from_2_to_3=router.get_json_format(
                    graph_json['charts']['weekly_cohort_from_2_to_3']),
                weekly_cohort_from_3_to_4=router.get_json_format(
                    graph_json['charts']['weekly_cohort_from_3_to_4']),
                data_type=data_type,
                filters=filters
            )

        if template == 'stats-purchase.html':
            graph_json, data_type, filters = router.get_chart(target='stats', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   daily_orders=router.get_json_format(graph_json['charts']['daily_orders']),
                                   weekly_orders=router.get_json_format(graph_json['charts']['weekly_orders']),
                                   monthly_orders=router.get_json_format(graph_json['charts']['monthly_orders']),
                                   hourly_orders=router.get_json_format(graph_json['charts']['hourly_orders']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'stats-desc.html':
            graph_json, data_type, filters = router.get_chart(target='descriptive', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   weekly_average_session_per_user=router.get_json_format(
                                       graph_json['charts']['weekly_average_session_per_user']),
                                   weekly_average_order_per_user=router.get_json_format(
                                       graph_json['charts']['weekly_average_order_per_user']),
                                   purchase_amount_distribution=router.get_json_format(
                                       graph_json['charts']['purchase_amount_distribution']),
                                   weekly_average_payment_amount=router.get_json_format(
                                       graph_json['charts']['weekly_average_payment_amount']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'abtest-promotion.html':
            graph_json, data_type, filters = router.get_chart(target='abtest-promotion', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   o_pa_diff=router.get_json_format(
                                       graph_json['charts']['order_and_payment_amount_differences']),
                                   promotion_comparison=router.get_json_format(
                                       graph_json['charts']['promotion_comparison']),
                                   promo_use_ba_a_accept=router.get_json_format(
                                       graph_json['charts']['promotion_usage_before_after_amount_accept']),
                                   promo_use_ba_a_reject=router.get_json_format(
                                       graph_json['charts']['promotion_usage_before_after_amount_reject']),
                                   promo_use_ba_o_accept=router.get_json_format(
                                       graph_json['charts']['promotion_usage_before_after_orders_accept']),
                                   promo_use_ba_o_reject=router.get_json_format(
                                       graph_json['charts']['promotion_usage_before_after_orders_reject']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'abtest-product.html':
            graph_json, data_type, filters = router.get_chart(target='abtest-product', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   product_use_ba_a_accept=router.get_json_format(
                                       graph_json['charts']['product_usage_before_after_amount_accept']),
                                   product_use_ba_a_reject=router.get_json_format(
                                       graph_json['charts']['product_usage_before_after_amount_reject']),
                                   product_use_ba_o_accept=router.get_json_format(
                                       graph_json['charts']['product_usage_before_after_orders_accept']),
                                   product_use_ba_o_reject=router.get_json_format(
                                       graph_json['charts']['product_usage_before_after_orders_reject']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'abtest-segments.html':
            graph_json, data_type, filters = router.get_chart(target='abtest-segments', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   sc_weekly_ba_orders=router.get_json_format(
                                       graph_json['charts']['segments_change_weekly_before_after_orders']),
                                   sc_daily_ba_orders=router.get_json_format(
                                       graph_json['charts']['segments_change_daily_before_after_orders']),
                                   sc_monthly_ba_orders=router.get_json_format(
                                       graph_json['charts']['segments_change_monthly_before_after_orders']),
                                   sc_weekly_ba_amount=router.get_json_format(
                                       graph_json['charts']['segments_change_weekly_before_after_amount']),
                                   sc_daily_ba_amount=router.get_json_format(
                                       graph_json['charts']['segments_change_daily_before_after_amount']),
                                   sc_monthly_ba_amount=router.get_json_format(
                                       graph_json['charts']['segments_change_monthly_before_after_amount']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'product.html':
            graph_json, data_type, filters = router.get_chart(target='product_analytic', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   most_combined_products=router.get_json_format(
                                       graph_json['charts']['most_combined_products']),
                                   most_ordered_products=router.get_json_format(
                                       graph_json['charts']['most_ordered_products']),
                                   most_ordered_categories=router.get_json_format(
                                       graph_json['charts']['most_ordered_categories']),
                                   data_type=data_type,
                                   filters=filters)
        if template == 'rfm.html':
            graph_json, data_type, filters = router.get_chart(target='rfm', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   rfm=router.get_json_format(graph_json['charts']['rfm']),
                                   frequency_recency=router.get_json_format(graph_json['charts']['frequency_recency']),
                                   monetary_frequency=router.get_json_format(graph_json['charts']['monetary_frequency']),
                                   recency_monetary=router.get_json_format(graph_json['charts']['recency_monetary']),
                                   data_type=data_type,
                                   filters=filters
                                   )

        if template == 'customer-segmentation.html':
            graph_json, data_type, filters = router.get_chart(target='customer-segmentation', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   segmentation=router.get_json_format(graph_json['charts']['segmentation']),
                                   frequency_clusters=router.get_json_format(graph_json['charts']['frequency_clusters']),
                                   monetary_clusters=router.get_json_format(graph_json['charts']['monetary_clusters']),
                                   recency_clusters=router.get_json_format(graph_json['charts']['recency_clusters']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'index2.html':
            graph_json, data_type, filters = router.get_chart(target='index2', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   rfm=router.get_json_format(
                                       graph_json['charts']['rfm']),
                                   purchase_amount_distribution=router.get_json_format(
                                       graph_json['charts']['purchase_amount_distribution']),
                                   uoc_order_seq=router.get_json_format(
                                       graph_json['charts']['user_counts_per_order_seq']),
                                   daily_funnel=router.get_json_format(
                                       graph_json['charts']['daily_funnel']),
                                   weekly_cohort_downloads=router.get_json_format(
                                       graph_json['charts']['weekly_cohort_downloads']),
                                   daily_clv=router.get_json_format(
                                       graph_json['charts']['daily_clv']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'clv.html':
            graph_json, data_type, filters = router.get_chart(target='clv', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   daily_clv=router.get_json_format(
                                       graph_json['charts']['daily_clv']),
                                   clvsegments_amount=router.get_json_format(
                                       graph_json['charts']['clvsegments_amount']),
                                   data_type=data_type,
                                   filters=filters)

        if template == 'anomaly.html':
            graph_json, data_type, filters = router.get_chart(target='anomaly', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   dfunnel_anomaly=router.get_json_format(
                                       graph_json['charts']['dfunnel_anomaly']),
                                   dcohort_anomaly=router.get_json_format(
                                       graph_json['charts']['dcohort_anomaly']),
                                   dcohort_anomaly_2=router.get_json_format(
                                       graph_json['charts']['dcohort_anomaly_2']),
                                   dorders_anomaly=router.get_json_format(
                                       graph_json['charts']['dorders_anomaly']),
                                   clvrfm_anomaly=router.get_json_format(
                                       graph_json['charts']['clvrfm_anomaly']),
                                   data_type=data_type,
                                   filters=filters)
        if template == 'delivery.html':
            graph_json, data_type, filters = router.get_chart(target='delivery', index=index, date=date)
            return render_template(template,
                                   segment=segment,
                                   ride=router.get_json_format(
                                       graph_json['charts']['ride']),
                                   deliver=router.get_json_format(
                                       graph_json['charts']['deliver']),
                                   prepare=router.get_json_format(
                                       graph_json['charts']['prepare']),
                                   prepare_weekday_hour=router.get_json_format(
                                       graph_json['charts']['prepare_weekday_hour']),
                                   deliver_weekday_hour=router.get_json_format(
                                       graph_json['charts']['deliver_weekday_hour']),
                                   ride_weekday_hour=router.get_json_format(
                                       graph_json['charts']['ride_weekday_hour']),
                                   kpis=graph_json['kpis'],
                                   data_type=data_type,
                                   filters=filters)

        if template not in ['funnel-customer.html', 'funnel-customer.html', 'index.html', 'index2.html', 'rfm.htm',
                            'product.html', 'abtest-segments.html', 'abtest-product.html', 'abtest-promotion.html',
                            'stats-desc.html', 'stats-purchase.htm', 'cohorts.html', 'customer-segmentation.html']:
            if template == 'delivery.html':
                router.add_new_message(dict(request.form))
                args = router.fetch_chats()
                return render_template(template,
                                       segment=segment,
                                       messages=args['messages'],
                                       chart=args['charts'],
                                       filters=args['filters']
                                       )
            else:
                router.execute_request(req=dict(request.form), template=segment)
                router.fetch_results(segment)
                values = router.message
                return render_template(template, segment=segment, values=values)

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    except Exception as e:
        print(e)
        return render_template('page-500.html'), 500



