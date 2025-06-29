# database.py
from flask_sqlalchemy import SQLAlchemy
import json

db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Accountgegevens
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    # Alle persoonlijke en medische gegevens
    woman_name = db.Column(db.String(120), nullable=False)
    partner_name = db.Column(db.String(120), nullable=True)
    woman_dob = db.Column(db.Date, nullable=True)
    due_date = db.Column(db.Date, nullable=False)
    midwifery_practice = db.Column(db.String(120), nullable=True)
    midwifery_phone = db.Column(db.String(20), nullable=True)
    woman_phone = db.Column(db.String(20), nullable=True)
    partner_phone = db.Column(db.String(20), nullable=True)
    baby_name = db.Column(db.String(120), nullable=True)
    baby_name_secret = db.Column(db.Boolean, default=False)
    medical_complications = db.Column(db.Text, nullable=True)

    # Relatie met het geboorteplan
    birth_plan = db.relationship('BirthPlan', backref='user', uselist=False, cascade="all, delete-orphan")


class BirthPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)

    # State wordt opgeslagen als JSON strings
    _plan_json = db.Column('plan', db.Text, nullable=False)
    _history_json = db.Column('history', db.Text, nullable=False)

    @property
    def plan(self):
        return json.loads(self._plan_json)

    @plan.setter
    def plan(self, value):
        self._plan_json = json.dumps(value)

    @property
    def history(self):
        return json.loads(self._history_json)

    @history.setter
    def history(self, value):
        self._history_json = json.dumps(value)

# De 'Sessions' class is hier bewust verwijderd.
# De Flask-Session extensie beheert deze tabel nu volledig zelf.

