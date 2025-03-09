from flask import Flask

from customeranalytics.app.config import ProductionConfig, DebugConfig
from customeranalytics.app.base.routes import blueprint as base_blueprint
from customeranalytics.app.home.routes import blueprint as home_blueprint


class CreateApp:
    def __init__(
            self,
            config: ProductionConfig | DebugConfig
    ):
        self.app = Flask(__name__, static_folder='base/static')
        self.config = config
        self.create_app()

    def register_blueprints(self):
        self.app.register_blueprint(base_blueprint)
        self.app.register_blueprint(home_blueprint)

    def create_app(self):
        self.app.config.from_object(self.config)
        self.register_blueprints()
