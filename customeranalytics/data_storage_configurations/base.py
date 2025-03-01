from flask_login import current_user

from customeranalytics.data_storage_configurations.sqlite import SQLiteDB


class BaseDataStorageConfiguration(SQLiteDB):
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
        logs table in sqlite table is updated.
        chats table in sqlite table is updated.
        """
        self.check_for_table_exits(table='logs')
        self.check_for_table_exits(table='chat')

        try:
            logs['login_user'] = current_user
            logs['log_time'] = str(self.current_date_to_day())[0:19]
            logs['general_message'] = logs['info']
            self.execute_query(
                self.insert_query(
                    table='logs',
                    columns=self.sqlite_queries['columns']['logs'][1:],
                    values=logs
                )
            )
        except Exception as e:
            print(e)

        try:
            self.execute_query(
                self.insert_query(
                    table='chat', columns=self.sqlite_queries['columns']['chat'][1:],
                     values=self.info_logs_for_chat(logs['info'])
                )
            )
        except Exception as e: print(e)