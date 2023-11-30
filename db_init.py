from main import app, db  # assuming your Flask app and db are defined in your_flask_app.py

with app.app_context():
    db.create_all()
