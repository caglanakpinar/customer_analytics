import os
from decouple import config


class Config(object):
    basedir = os.path.abspath(os.path.dirname(__file__))
    SECRET_KEY = config('SECRET_KEY', default='S#perS3crEt_007')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'db.sqlite3')
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    DEBUG = False


class DebugConfig(Config):
    DEBUG = True


config_dict = {'Production': ProductionConfig, 'Debug': DebugConfig}
