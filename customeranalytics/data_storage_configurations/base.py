from flask_login import current_user

from customeranalytics.data_storage_configurations.connection import Connection


class BaseDataStorageConfiguration(Connection):
    def __init__(self):
        super().__init__()
        self.info_logs_for_chat = lambda info: {'user': 'info',
                                                'date': str(self.current_date_to_day())[0:19],
                                                'user_logo': 'info.jpeg',
                                                'chat_type': 'info', 'chart': '',
                                                'general_message': info, 'message': ''}

    def info_logs_for_chat(self, info):
        return {
            'user': 'info',
            'date': str(self.current_date_to_day())[0:19],
            'user_logo': 'info.jpeg',
            'chat_type': 'info',
            'chart': '',
            'general_message': info,
            'message': ''
        }

    def suc_log_for_data_works(self, x, e):
        return f"{self.DATA_WORKS_READABLE_FORM[x]} is created! Check {self.DATA_WORKS_READABLE_FORM[x]} sections. "

    def fail_log_for_data_works(self, x, e):
        return f"{self.DATA_WORKS_READABLE_FORM[x]} is failed! Check {self.DATA_WORKS_READABLE_FORM[x]} sections. - {e}"

    def logs_update(self, logs):
        """
        logs in connection.yaml is updated. it has been updated on given temp folder
        """
        logs['log_time'] = str(self.current_date_to_day())[0:19]
        logs['general_message'] = logs['info']
        self.update_logs(logs)
