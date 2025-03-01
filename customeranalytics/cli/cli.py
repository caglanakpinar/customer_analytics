import click
from flask_login import LoginManager
from flask_migrate import Migrate


from customeranalytics.app import CreateApp
from customeranalytics.app.config import config_dict
from customeranalytics.data_storage_configurations import SQLiteDB


login_manager = LoginManager()
sqlite_db = SQLiteDB()


@click.group()
def cli():
    pass


@cli.group(name="interface")
def interface_run():
    pass


@interface_run.commmand(
    name='app',
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--port", required=False)
@click.option("--host", required=False)
@click.option("--debug", required=False,
   default=False,
)
def app(
        post=None,
        host=None,
        debug=False
):

    app = CreateApp(
        sqlite_db.db,
        login_manager,
        config_dict[('Debug'if debug else 'Production')]
    )

    Migrate(app.app, sqlite_db.db)
    app.app.run(
        port=post,
        host=host
    )


def create_user_interface():
    """
    This process triggers web interface
    port: port is stored at web_interface.yaml
    host: host is stored at web_interface.yaml
    """
    path = join(r.abspath_for_sample_data(), "web", "run.py")
    cmd = "python " + path
    print('http://' + str(web_configs['host']) + ':' + str(web_configs['port']))
    _p = subprocess.Popen(cmd, shell=True)