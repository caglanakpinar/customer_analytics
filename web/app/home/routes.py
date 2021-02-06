import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from web.app.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound

from web.app.home.models import RouterRequest
from web.app.home.forms import SampleData, RealData, Charts, charts


samples = SampleData()
real = RealData()
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
    charts.get_chart(target='index') # collect charts on index.html
    return render_template('index.html',
                           segment='index',
                           charts=charts.graph_json['charts']['orders_daily'],
                           customer_segments=charts.graph_json['charts']['customer_segmentation'],
                           kpis=charts.graph_json['kpis']
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

        if template == 'index2.html':
            charts.get_chart(target='index2')
            return render_template(template, segment=segment, rfm=charts.graph_json['charts']['rfm'])
        else:
            reqs.execute_request(req=dict(request.form), template=segment)
            values = reqs.fetch_results(segment)
            return render_template(template, segment=segment, values=values)

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500



