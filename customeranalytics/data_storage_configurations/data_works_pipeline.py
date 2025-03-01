import threading
from flask_login import current_user

from customeranalytics.data_storage_configurations.base import BaseDataStorageConfiguration
from customeranalytics.exploratory_analysis import create_exploratory_analyse
from customeranalytics.ml_process import create_ml
from customeranalytics.data_storage_configurations.es_create_index import CreateIndex
from customeranalytics.data_storage_configurations.query_es import QueryES


class DataPipelines(BaseDataStorageConfiguration):
    def __init__(self, es_tag, data_connection_structure, ea_connection_structure, ml_connection_structure,
                 data_columns, actions):
        """
        :param es_tag: elasticsearch tag name that is created on web interface 'ElasticSearch Configuration' page
        :param data_connection_structure: check :data_configs above
        :param ea_connection_structure: check :ea_configs above
        :param ml_connection_structure: check :ml_configs above
        :param data_columns: check :data_columns above
        :param actions: whole actions which are stored into the actions table in sqlite
        """
        super().__init__()
        self.es_tag = es_tag
        self.data_connection_structure = data_connection_structure
        self.ea_connection_structure = ea_connection_structure
        self.ml_connection_structure = ml_connection_structure
        self.actions = actions
        self.data_columns = data_columns
        self.es_con = self.collect_data_from_table(table="es_connection", return_list=True)[-1]
        self.create_index = CreateIndex(data_connection_structure=data_connection_structure,
                                        data_columns=data_columns, actions=actions)
        self.query_es = QueryES(host=self.es_con['host'], port=self.es_con['port'])
        self.unique_dimensions = []
        self.schedule = True
        self.separator = lambda dim: [print("*" * 20) for i in range(3)] + [
            print("*" * 10, " ", " DIMENSION : ", dim, "*" * 10)]
        self.suc_log_for_ea = """
            Exploratory Analysis are Created! Check Funnel, Cohort, 
            Descriptive Stats sections.
        """

    def info_log_create(self, type, dim=None):
        """
        Logging system on creating E.A. or ML Works. This print processes are also shown at profile.html activities.
        This gives the information of the process are done.
        """
        _info = self.suc_log_for_data_works(type)
        if dim is not None:
            _info += " || dimension : " + dim
        print("-- Process Info --")
        print(_info)
        self.logs_update(logs={"page": "data-execute", "info": _info, "color": "green"})

    def fail_log_create(self, e, type, dim=None):
        """
        Logging system on creating E.A. or ML Works. This print processes are also shown at profile.html activities.
        This gives the information of the process are failed.
        """
        e_str = ''
        if e is not None:
            e_str = str(e)
            e_str = e_str[:min(len(e_str), 100)]
        fail_message = self.fail_log_for_data_works(type, e_str)
        if dim is not None:
            fail_message += " || dimension : " + dim
        print(" FAIL ---- !!!!!!!!")
        print("-- message :", fail_message)
        print(" ----- description ::::::", e_str)
        self.logs_update(logs={"page": "data-execute",
                               "info": fail_message.replace("'", " "),
                               "color": "red"})

    def execute_pipe(self, _conf, execution, type, dim=None):
        try:
            execution(_conf, type)
            self.info_log_create(type=type)
        except Exception as e:
            print(e)
            self.fail_log_create(e=e, type=type, dim=dim)

    def pipe_1(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ml_connection_structure, create_ml, 'clv_prediction')

    def pipe_2(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'rfm', dim=dim)
        self.execute_pipe(ml_connection_structure, create_ml, 'segmentation', dim=dim)

    def pipe_3(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'funnel', dim=dim)
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'cohort', dim=dim)

    def pipe_4(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'stats', dim=dim)
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'churn', dim=dim)

    def pipe_5(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ml_connection_structure, create_ml, 'abtest', dim=dim)
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'products', dim=dim)
        self.execute_pipe(ea_connection_structure, create_exploratory_analyse, 'promotions', dim=dim)

    def pipe_6(self, ml_connection_structure, ea_connection_structure, dim=None):
        self.execute_pipe(ml_connection_structure, create_ml, 'anomaly', dim=dim)
        self.execute_pipe(ml_connection_structure, create_ml, 'delivery_anomaly', dim=dim)

    def data_work_pipelines_execution(self, ml_connection_structure, ea_connection_structure, dim=None):
        _kwargs = {"ml_connection_structure": ml_connection_structure,
                   'ea_connection_structure': ea_connection_structure, 'dim': dim}
        if dim is None:
            self.pipe_1(**_kwargs)
        self.pipe_2(**_kwargs)
        pipes = [self.pipe_3, self.pipe_4]
        for pipe in pipes:
            process = threading.Thread(target=pipe, kwargs=_kwargs)
            process.daemon = True
            process.start()
        process.join()
        self.pipe_5(**_kwargs)
        self.pipe_6(**_kwargs)
