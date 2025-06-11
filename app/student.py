from flask import Blueprint, render_template, session, current_app, request, redirect, jsonify, url_for
from app.utility.common_util import immutable_to_dict

student_bp = Blueprint('student', __name__, url_prefix='/student/')

@student_bp.route('/login/', methods=['POST'])
def login_user():
    data = immutable_to_dict(request.form)
    username = data['uname']
    password = data['password']
    user = current_app.config["users"].find_one({'uname': username})

    if user and password == user["password"]:# and hashpw(password.encode('utf-8'), user['password']) == user['password']:
        # Login successful, store user_id in session
        session['user_id'] = str(user['_id'])
        print(session)
        #current_app.config["sid"] = str(user['_id'])
        return redirect(url_for("student.home"))
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

@student_bp.route("/")
def home():
    return render_template("student/home.html",
                           uid=session.get("user_id"))

@student_bp.route("/reports")
def reports():
    return render_template("student/reports.html")