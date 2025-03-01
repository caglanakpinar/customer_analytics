from flask_wtf import FlaskForm
from wtforms import TextAreaField, PasswordField
from wtforms.validators import InputRequired, Email, DataRequired


class LoginForm(FlaskForm):
    username = TextAreaField('Username', id='username_login', validators=[DataRequired()])
    password = PasswordField('Password', id='pwd_login', validators=[DataRequired()])


class CreateAccountForm(FlaskForm):
    username = TextAreaField('Username', id='username_create', validators=[DataRequired()])
    email= TextAreaField('Email', id='email_create', validators=[DataRequired(), Email()])
    password = PasswordField('Password', id='pwd_create', validators=[DataRequired()])


