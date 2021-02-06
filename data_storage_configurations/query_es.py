from elasticsearch import Elasticsearch
from elasticsearch import helpers
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from configs import default_es_port, default_es_host, elasticsearch_settings, elasticsearch_settings_reports


class QueryES:
    def __init__(self, host=None, port=None):
        """
        :param host: elasticsearch host
        :param port: elasticsearch port
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.es = Elasticsearch([{'host': self.host, 'port': self.port}], timeout=3000, max_retries=100)
        self.match = {}
        self.query_size = elasticsearch_settings['settings']['index.query.default_field']
        self.fields = False
        self.source = False
        self.date_queries = []
        self.boolean_queries = []

    def date_queries_builder(self, expression):
        self.date_queries = [{"range": expression}]

    def boolean_queries_buildier(self, expression):
        self.boolean_queries = [{"term": expression}]

    def query_builder(self, fields, boolean_queries=None, date_queries=None, _source=False):
        """
        creates elasticsearch queries.
        In order to query ea, make sure data has been stored properly
        :param fields: query fields expected from the returned query
        :param boolean_queries: if there is True/False query
        :param date_queries: if there is date format queries
        :param _source: if need to retun whole filtered index object
        :return: query dictionary
        """

        self.match = {
                        "size": self.query_size, "from": 0,
                        "_source": False if not _source else True
                     }
        if fields is not None:
            self.fields = True
            self.match["fields"] = fields

        if _source:
            self.source = True

        if boolean_queries is not None:
            self.boolean_queries = boolean_queries

        if date_queries is not None:
            self.date_queries = date_queries

        self.match['query'] = {"bool": {"must": self.boolean_queries + self.date_queries}}
        print(self.match)

    def get_data_from_es(self, index='orders'):
        """

        :return:
        """
        res = []
        for r in self.es.search(index=index, body=self.match)['hits']['hits']:
            _obj = {}
            if self.fields:
                _obj = {f: r['fields'][f][0] for f in r['fields']}
            if self.source:
                _obj['_source'] = r['_source']
            res.append(_obj if _obj != {} else r)
        return res

    def get_insert_obj(self, list_of_obj, index):
        for i in list_of_obj:
            add_cmd = {"_index": index,
                       "_id": i['id'],
                       "_source": i}
            yield add_cmd

    def create_index(self, index):
        try:
            self.es.indices.create(index, body=elasticsearch_settings_reports)
        except Exception as e:
            print("index already exists !!!")

    def check_index_exists(self, index):
        if self.es.indices.exists(index=index):
            return True
        else:
            self.create_index(index=index)

    def insert_data_to_index(self, list_of_obj, index):
        self.check_index_exists(index=index)
        helpers.bulk(self.es, self.get_insert_obj(list_of_obj, index))












