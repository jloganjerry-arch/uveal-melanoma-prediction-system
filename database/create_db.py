from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sex = db.Column(db.Integer, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    race = db.Column(db.Integer, nullable=False)
    stage = db.Column(db.Integer, nullable=False)
    radiation = db.Column(db.Integer, nullable=False)
    chemo = db.Column(db.Integer, nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    risk_category = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.id}>'

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
