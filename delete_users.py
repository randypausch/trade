from main import app, db, User

with app.app_context():
    # Delete all users from the User table
    db.session.query(User).delete()
    db.session.commit()