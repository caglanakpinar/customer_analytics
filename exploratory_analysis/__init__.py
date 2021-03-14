import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from exploratory_analysis.funnels import Funnels
from exploratory_analysis.cohorts import Cohorts
from exploratory_analysis.product_analytics import ProductAnalytics
from exploratory_analysis.rfm import RFM
from exploratory_analysis.descriptive_statistics import Stats


configs = {"date": None,
           "funnel": {"actions": ["download", "signup"],
                      "purchase_actions": ["has_basket", "order_screen"],
                      "host": 'localhost',
                      "port": '9200',
                      'download_index': 'downloads',
                      'order_index': 'orders'},
           "cohort": {"has_download": True, "host": 'localhost', "port": '9200'},
           "products": {"has_download": True, "host": 'localhost', "port": '9200'},
           "rfm": {"host": 'localhost', "port": '9200', 'download_index': 'downloads', 'order_index': 'orders'},
           "stats": {"host": 'localhost', "port": '9200', 'download_index': 'downloads', 'order_index': 'orders'}
          }


schedule_configs = {"date": None,
           "funnel": {"actions": [],
                      "purchase_actions": [],
                      "host": 'localhost',
                      "port": '9200',
                      'download_index': 'downloads',
                      'order_index': 'orders'},
           "cohort": {"has_download": True, "host": 'localhost', "port": '9200'},
           "products": {"has_download": True, "host": 'localhost', "port": '9200'},
           "rfm": {"host": 'localhost', "port": '9200', 'download_index': 'downloads', 'order_index': 'orders'},
           "stats": {"host": 'localhost', "port": '9200', 'download_index': 'downloads', 'order_index': 'orders'}
          }

exploratory_analysis = {'funnel': Funnels,
                        'cohort': Cohorts,
                        'products': ProductAnalytics,
                        'rfm': RFM,
                        'stats': Stats}


def create_exploratory_analysis(configs):
    ea = {a: exploratory_analysis[a](**configs[a]) for a in exploratory_analysis}
    ea['funnel'].purchase_action_funnel(start_date=configs['date'])
    ea['funnel'].download_signup_session_order_funnel(start_date=configs['date'])
    ea['funnel'].overall_funnel(start_date=configs['date'])
    ea['cohort'].execute_cohort(start_date=configs['date'])
    ea['rfm'].execute_rfm(start_date=configs['date'])
    ea['stats'].execute_descriptive_stats(start_date=configs['date'])


def query_exploratory_analysis(configs, queries, ea):
    ea = exploratory_analysis[ea](**configs[ea])
    return ea.fetch(**queries)

