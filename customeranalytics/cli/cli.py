import click
from flask_migrate import Migrate


from customeranalytics.app.create_app import CreateApp
from customeranalytics.app.config import config_dict
from customeranalytics.data_storage_configurations import SQLiteDB

sqlite_db = SQLiteDB()


@click.group()
def cli():
    pass


@cli.group(name="interface")
def interface_run():
    pass


@interface_run.command(
    name='app',
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--port", required=True)
@click.option("--host", required=True)
@click.option("--debug", required=True, type=click.BOOL)
def app(
        port=None,
        host=None,
        debug=False
):

    app = CreateApp(
        sqlite_db.db,
        config_dict[('Debug'if debug else 'Production')]
    )

    Migrate(app.app, sqlite_db.db)
    app.app.run(
        port=port,
        host=host,
        debug=debug
    )


if __name__ == '__main__':
    cli()