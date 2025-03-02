from flask_sqlalchemy import SQLAlchemy
from flask import Flask

from customeranalytics.app.config import ProductionConfig, DebugConfig
from customeranalytics.app.base.routes import blueprint as base_blueprint
from customeranalytics.app.home.routes import blueprint as home_blueprint


class CreateApp:
    def __init__(
            self,
            db: SQLAlchemy,
            config: ProductionConfig | DebugConfig
    ):
        self.db = db
        self.app = Flask(__name__, static_folder='base/static')
        self.config = config
        self.create_app()

    def register_extensions(self, ):
        self.db.init_app(self.app)

    def register_blueprints(self):
        self.app.register_blueprint(base_blueprint)
        self.app.register_blueprint(home_blueprint)

    def configure_database(self):
        @self.app.before_request
        def initialize_database():
            self.db.create_all()

        @self.app.teardown_request
        def shutdown_session(exception=None):
            self.db.session.remove()

    def create_app(self):
        self.app.config.from_object(self.config)
        self.register_extensions()
        self.register_blueprints()
        self.configure_database()
