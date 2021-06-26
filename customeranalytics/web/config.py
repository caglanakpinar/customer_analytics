import os
from decouple import config

try: from utils import read_yaml, abspath_for_sample_data
except: from customeranalytics.utils import read_yaml, abspath_for_sample_data


class Config(object):
    basedir = os.path.abspath(os.path.dirname(__file__))
    SECRET_KEY = config('SECRET_KEY', default='S#perS3crEt_007')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'db.sqlite3')
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    DEBUG = False
    ALLOWED_IMAGE_EXTENSIONS = ["JPEG", "JPG", "PNG", "GIF"]
    MAX_IMAGE_FILESIZE = 0.5 * 1024 * 1024


class DebugConfig(Config):
    DEBUG = True
    ALLOWED_IMAGE_EXTENSIONS = ["JPEG", "JPG", "PNG", "GIF"]
    MAX_IMAGE_FILESIZE = 0.5 * 1024 * 1024


config_dict = {'Production': ProductionConfig, 'Debug': DebugConfig}
web_configs = read_yaml(os.path.join(abspath_for_sample_data(), "docs"), "web_configs.yaml")


