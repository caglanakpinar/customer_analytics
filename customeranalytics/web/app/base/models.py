from flask_login import UserMixin
from sqlalchemy import Column, Integer, String

try:
    from sqlalchemy import Binary
except:
    from sqlalchemy import LargeBinary as Binary


try: from web.app import db, login_manager
except: from customeranalytics.web.app import db, login_manager

try: from web.app.base.util import hash_pass
except: from customeranalytics.web.app.base.util import hash_pass


class User(db.Model, UserMixin):
    __tablename__ = 'User'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password = Column(Binary)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            if hasattr(value, '__iter__') and not isinstance(value, str):
                value = value[0]

            if property == 'password':
                value = hash_pass(value)
            setattr(self, property, value)

    def __repr__(self):
        return str(self.username)


@login_manager.user_loader
def user_loader(id):
    return User.query.filter_by(id=id).first()


@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    user = User.query.filter_by(username=username).first()
    return user if user else None
