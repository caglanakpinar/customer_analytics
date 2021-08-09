from elasticsearch import Elasticsearch
from elasticsearch import helpers
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_es_port, default_es_host, elasticsearch_settings, elasticsearch_settings_reports
from customeranalytics.configs import default_es_bulk_insert_chunk_size, default_es_bulk_insert_chunk_bytes, max_elasticsearch_bulk_insert_bytes


class QueryES:
    """
    This is for querying elasticsearch.
    Mostly, it is using for building elastic search query insert data into the index.
    """
    def __init__(self, host=None, port=None):
        """
        query_size: default query size from configs.py.

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
        """
        date related filtering on ElasticSearch queries.

        :param expression: e.g. {'session_start_date': {'gte': '2021-01-01T00:00:00'}
        """
        self.date_queries = [{"range": expression}]

    def boolean_queries_buildier(self, expression):
        """
        Boolean filed (True/False) filtering.
        There are actions and has_purchased column which are stored in ElasticSearch Index with the boolean format.

        :param expression: e.g. {"actions.has_purchased": True}
        """
        self.boolean_queries = [{"term": expression}]

    def query_builder(self, fields, boolean_queries=None, date_queries=None, _source=False):
        """
        creates elasticsearch queries.
        In order to query ea, make sure data has been stored properly.

        match = {"size": self.query_size, "from": 0
                 "_source": False (If it is 'True', it will not be easy to query)
                 "fields": ["session_start_date", "client"], comes with arguments,
                 "query": {"bool": {"must": [{"term": {...boolean queries ..}},
                                             {"term": {...date queries ..}}
                                                ]}
                }

        :param fields: query fields expected from the returned query
        :param boolean_queries: if there is True/False query
        :param date_queries: if there are date format queries
        :param _source: if need to return whole filtered index object
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

    def get_data_from_es(self, index='orders'):
        """
        query ElasticSearch index by using self.match.
        :return: list of object
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
        """
        bulk insert into the given index.
        :param list_of_obj: bulk inserting data
        :param index: index name (downloads, orders, ..)
        """
        for i in list_of_obj:
            add_cmd = {"_index": index,
                       "_id": i['id'],
                       "_source": i}
            yield add_cmd

    def create_index(self, index):
        """
        If the index has not been created, yet, This can handle the creation of the index task.
        :param index: index name for the creation
        """
        try: self.es.indices.create(index, body=elasticsearch_settings_reports)
        except: print("index already exists !!!")

    def check_index_exists(self, index):
        """
        Checking the recent list of indexes in the given host and port.
        If the index has not been created, yet, directly send it to the create_index
        :param index: checking index name
        """
        if self.es.indices.exists(index=index):
            return True
        else:
            self.create_index(index=index)

    def insert_data_to_index(self, list_of_obj, index):
        """
        bulk insert into the given index. Before inserting, checking if indexes exist.
        :param list_of_obj: bulk inserting data
        :param index: index name (downloads, orders, ..)
        """
        self.check_index_exists(index=index)
        helpers.bulk(self.es, self.get_insert_obj(list_of_obj, index),
                     max_chunk_bytes=default_es_bulk_insert_chunk_bytes,
                     chunk_size=default_es_bulk_insert_chunk_size)













