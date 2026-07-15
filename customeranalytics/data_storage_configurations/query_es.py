import sys, os, inspect
import json
import sqlite3
import datetime
from os.path import join, exists, dirname, abspath

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from customeranalytics.configs import default_es_port, default_es_host, elasticsearch_settings, elasticsearch_settings_reports
from customeranalytics.configs import default_es_bulk_insert_chunk_size, default_es_bulk_insert_chunk_bytes, max_elasticsearch_bulk_insert_bytes
from customeranalytics.utils import abspath_for_sample_data, get_storage_config


# ----------------------------------------------------------------------------------------------------------------------
# Local (SQLite) storage backend
#
# CustomerAnalytics used to store every ingested document (orders/downloads/products/deliveries) and every computed
# report inside an ElasticSearch cluster the user had to install and run separately. The project only ever used ES as
# a JSON document store with term/range filtering + a single distinct-value aggregation, so it is replaced here with an
# embedded SQLite database (JSON1). No server to install, and the data never leaves the machine.
#
# Each ES "index" becomes a SQLite table  "{index}"(_id TEXT PRIMARY KEY, _source TEXT/JSON).  The ElasticSearch query
# DSL that the rest of the codebase builds (``{"size", "from", "_source", "fields", "sort", "query": {"bool": {"must":
# [{"term": ...}, {"range": ...}]}}, "aggs": {...}}``) is translated to SQL against ``json_extract`` so that QueryES and
# its ``.es`` shim remain a drop-in replacement for the previous ElasticSearch-backed implementation.
# ----------------------------------------------------------------------------------------------------------------------


def _json_default(o):
    """Serialize values pandas/numpy/datetime put into report + index documents."""
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    if hasattr(o, "item"):  # numpy scalar
        try:
            return o.item()
        except Exception:
            pass
    if isinstance(o, set):
        return list(o)
    return str(o)


def _coerce(value):
    """Coerce a filter value into something comparable with the JSON-stored (text/number) values."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value


def _json_path(field):
    """'actions.purchased' -> '$.actions.purchased', 'report_name' -> '$.report_name'."""
    return "$." + field


def _extract(source, field):
    """Navigate a possibly dotted field path inside an already decoded _source dict."""
    cur = source
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def resolve_data_db_path():
    """
    Location of the embedded analytics database.

    Storage requires no configuration: the database lives in the package folder alongside the application db.sqlite3.
    """
    directory = get_storage_config()['directory']
    if directory and exists(directory):
        return join(directory, "web", "customeranalytics_data.sqlite3")
    return join(abspath_for_sample_data(), "web", "customeranalytics_data.sqlite3")


# expression indexes created (best effort) to keep the common filters fast
_INDEX_HINTS = {
    "reports": ["$.index", "$.report_name", "$.report_date"],
    "orders": ["$.session_start_date", "$.client", "$.dimension", "$.actions.purchased"],
    "downloads": ["$.download_date", "$.client"],
    "products": ["$.order_id"],
    "deliveries": ["$.order_id"],
}


class LocalStorage:
    """Thin sqlite3 wrapper that speaks a subset of the ElasticSearch query DSL used across the project."""

    _connections = {}

    def __init__(self, path=None):
        self.path = path or resolve_data_db_path()
        self.con = self._get_connection(self.path)

    @classmethod
    def _get_connection(cls, path):
        con = cls._connections.get(path)
        if con is None:
            con = sqlite3.connect(path, check_same_thread=False, timeout=30)
            con.row_factory = sqlite3.Row
            try:
                con.execute("PRAGMA journal_mode=WAL;")
                con.execute("PRAGMA busy_timeout=30000;")
            except Exception:
                pass
            cls._connections[path] = con
        return con

    # -- schema ------------------------------------------------------------------------------------------------------
    def table_exists(self, index):
        row = self.con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (index,)
        ).fetchone()
        return row is not None

    def create_table(self, index):
        self.con.execute(
            'CREATE TABLE IF NOT EXISTS "{}" (_id TEXT PRIMARY KEY, _source TEXT)'.format(index)
        )
        for i, path in enumerate(_INDEX_HINTS.get(index, [])):
            try:
                # SQLite forbids bound parameters inside a CREATE INDEX expression; the json path comes from the
                # internal _INDEX_HINTS constant (never user input) so it is inlined directly.
                self.con.execute(
                    'CREATE INDEX IF NOT EXISTS "{idx}_{i}" ON "{idx}" '
                    "(json_extract(_source, '{path}'))".format(idx=index, i=i, path=path)
                )
            except Exception as e:
                print("could not create expression index on {}: {}".format(index, e))
        self.con.commit()

    def count(self, index):
        if not self.table_exists(index):
            return 0
        return int(self.con.execute('SELECT COUNT(*) FROM "{}"'.format(index)).fetchone()[0])

    # -- writes ------------------------------------------------------------------------------------------------------
    def bulk_insert(self, index, list_of_obj):
        if not self.table_exists(index):
            self.create_table(index)
        rows = []
        for obj in list_of_obj:
            _id = str(obj.get("id"))
            rows.append((_id, json.dumps(obj, default=_json_default)))
        self.con.executemany(
            'INSERT INTO "{}" (_id, _source) VALUES (?, ?) '
            'ON CONFLICT(_id) DO UPDATE SET _source=excluded._source'.format(index),
            rows,
        )
        self.con.commit()

    # -- reads -------------------------------------------------------------------------------------------------------
    def _build_where(self, query):
        """Translate {'bool': {'must': [{'term': {...}}, {'range': {...}}]}} into a SQL where clause + params."""
        clauses, params = [], []
        if not query:
            return "", params
        must = query.get("bool", {}).get("must", [])
        for cond in must:
            if "term" in cond:
                for field, value in cond["term"].items():
                    clauses.append("json_extract(_source, ?) = ?")
                    params.extend([_json_path(field), _coerce(value)])
            elif "range" in cond:
                for field, ops in cond["range"].items():
                    for op, value in ops.items():
                        sql_op = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<"}.get(op)
                        if sql_op is None:
                            continue
                        clauses.append("json_extract(_source, ?) {} ?".format(sql_op))
                        params.extend([_json_path(field), _coerce(value)])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def search(self, index, body):
        """ElasticSearch-compatible search returning {'hits': {'hits': [...]}, ('aggregations': {...})}."""
        result = {"hits": {"hits": []}}
        if not self.table_exists(index):
            return result

        # aggregations: only the distinct-value ("terms") aggregation is used across the project
        aggs = body.get("aggs")
        if aggs:
            result["aggregations"] = {}
            for agg_name, agg_body in aggs.items():
                terms = agg_body.get("terms", {})
                field = terms.get("field", "").replace(".keyword", "")
                size = terms.get("size", 10)
                sql = (
                    'SELECT json_extract(_source, ?) AS k, COUNT(*) AS c FROM "{}" '
                    "WHERE k IS NOT NULL GROUP BY k ORDER BY c DESC LIMIT ?".format(index)
                )
                buckets = [
                    {"key": r["k"], "doc_count": r["c"]}
                    for r in self.con.execute(sql, (_json_path(field), size)).fetchall()
                ]
                result["aggregations"][agg_name] = {"buckets": buckets}
            if body.get("size", 0) == 0:
                return result

        size = body.get("size", 10)
        offset = body.get("from", 0)
        fields = body.get("fields")
        where, params = self._build_where(body.get("query"))

        order = ""
        sort = body.get("sort")
        if sort:
            parts = []
            for col, direction in sort.items():
                direction = direction if isinstance(direction, str) else direction.get("order", "asc")
                parts.append("json_extract(_source, ?) {}".format("DESC" if str(direction).lower() == "desc" else "ASC"))
                params.append(_json_path(col))
            order = " ORDER BY " + ", ".join(parts)

        sql = 'SELECT _id, _source FROM "{}"{}{} LIMIT ? OFFSET ?'.format(index, where, order)
        params.extend([size, offset])

        for row in self.con.execute(sql, params).fetchall():
            source = json.loads(row["_source"])
            hit = {"_index": index, "_id": row["_id"], "_source": source}
            if fields:
                hit_fields = {}
                for f in fields:
                    value = _extract(source, f)
                    if value is not None:
                        hit_fields[f] = [value]
                hit["fields"] = hit_fields
            result["hits"]["hits"].append(hit)
        return result


class _Indices:
    """Emulates the ``client.indices`` namespace used by the codebase."""

    def __init__(self, storage):
        self._storage = storage

    def exists(self, index):
        return self._storage.table_exists(index)

    def create(self, index, body=None):
        self._storage.create_table(index)


class _Cat:
    """Emulates the ``client.cat`` namespace (only ``count`` is used)."""

    def __init__(self, storage):
        self._storage = storage

    def count(self, index, params=None):
        return [{"count": self._storage.count(index)}]


class _ESCompat:
    """
    Drop-in replacement for the small slice of the ElasticSearch client the project relied on:
    ``.search``, ``.indices.create/exists``, ``.cat.count`` and ``.ping``.
    """

    def __init__(self, storage):
        self._storage = storage
        self.indices = _Indices(storage)
        self.cat = _Cat(storage)

    def search(self, index=None, body=None, **kwargs):
        return self._storage.search(index, body or {})

    def ping(self):
        return True


class QueryES:
    """
    Build and run queries against the embedded analytics store.

    The public surface (``query_builder``, ``date_queries_builder``, ``boolean_queries_buildier``, ``get_data_from_es``,
    ``create_index``, ``check_index_exists``, ``insert_data_to_index`` and the ``.es`` client) is unchanged from the
    previous ElasticSearch-backed implementation, so callers do not need to know the data now lives in SQLite.
    """
    def __init__(self, host=None, port=None):
        """
        query_size: default query size from configs.py.

        :param host: kept for backwards compatibility (ignored by the local storage backend)
        :param port: kept for backwards compatibility (ignored by the local storage backend)
        """
        self.port = default_es_port if port is None else port
        self.host = default_es_host if host is None else host
        self.storage = LocalStorage()
        self.es = _ESCompat(self.storage)
        self.match = {}
        self.query_size = elasticsearch_settings['settings']['index.query.default_field']
        self.fields = False
        self.source = False
        self.date_queries = []
        self.boolean_queries = []

    def date_queries_builder(self, expression):
        """
        date related filtering on queries.

        :param expression: e.g. {'session_start_date': {'gte': '2021-01-01T00:00:00'}
        """
        self.date_queries = [{"range": expression}]

    def boolean_queries_buildier(self, expression):
        """
        Boolean filed (True/False) filtering.
        There are actions and has_purchased column which are stored with the boolean format.

        :param expression: e.g. {"actions.has_purchased": True}
        """
        self.boolean_queries = [{"term": expression}]

    def query_builder(self, fields, boolean_queries=None, date_queries=None, _source=False):
        """
        creates queries.
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
        query the index by using self.match.
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
        try: self.storage.create_table(index)
        except: print("index already exists !!!")

    def check_index_exists(self, index):
        """
        Checking the recent list of indexes in the given storage.
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
        self.storage.bulk_insert(index, list_of_obj)
