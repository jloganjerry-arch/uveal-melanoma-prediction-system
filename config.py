import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key_here'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'database/melanoma.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Model Risk Configuration Thresholds
    RISK_THRESHOLD_HIGH = 70.0
    RISK_THRESHOLD_INTERMEDIATE = 50.0
