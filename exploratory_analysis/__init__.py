import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from exploratory_analysis.funnels import Funnels
from exploratory_analysis.cohorts import Cohorts
from exploratory_analysis.product_analytics import ProductAnalytics
from exploratory_analysis.rfm import RFM
from exploratory_analysis.descriptive_statistics import Stats


ea_configs = {"date": None,
              "funnel": {"actions": ["download", "signup"],
                         "purchase_actions": ["has_basket", "order_screen"],
                         "host": 'localhost',
                         "port": '9200',
                         'download_index': 'downloads',
                         'order_index': 'orders'},
              "cohort": {"has_download": True, "host": 'localhost', "port": '9200',
                         'download_index': 'downloads', 'order_index': 'orders'},
              "products": {"has_product_connection": True, "host": 'localhost', "port": '9200',
                           "download_index": 'downloads', "order_index": 'orders'},
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
    print("*"*5, " Funnels ", "*"*5)
    ea['funnel'].purchase_action_funnel(start_date=configs['date'])
    ea['funnel'].download_signup_session_order_funnel(start_date=configs['date'])
    ea['funnel'].overall_funnel(start_date=configs['date'])
    del ea['funnel']
    print("*" * 5, " Cohorts ", "*" * 5)
    ea['cohort'].execute_cohort(start_date=configs['date'])
    del ea['cohort']
    print("*" * 5, " RFM ", "*" * 5)
    ea['rfm'].execute_rfm(start_date=configs['date'])
    del ea['rfm']
    print("*" * 5, " Descriptive Statistics ", "*" * 5)
    ea['stats'].execute_descriptive_stats(start_date=configs['date'])
    del ea['stats']
    print("*" * 5, " Product Analytics ", "*" * 5)
    ea['products'].execute_product_analysis(end_date=configs['date'])
    del ea


def query_exploratory_analysis(configs, queries, ea):
    ea = exploratory_analysis[ea](**configs[ea])
    return ea.fetch(**queries)

