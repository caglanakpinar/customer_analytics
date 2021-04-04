## TODO: this will be updated
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from flask_migrate import Migrate
from sys import exit
from decouple import config

try:
    from web.config import config_dict
    from web.app import create_app, db

except Exception as e:
    from .web.config import config_dict
    from .web.app import create_app, db


try:
    # Load the configuration using the default values 
    app_config = config_dict['Production']
except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)
Migrate(app, db)

if __name__ == "__main__":
    app.run()
