from flask import Flask, request, jsonify, session, redirect, current_app, Blueprint
from app.utility.common_util import immutable_to_dict

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/login/admin/', methods=['POST'])
def login_admin():
    data = immutable_to_dict(request.form)
    print(data)
    print(current_app.config["admin"])
    username = data['uname']
    password = data['password']

    # Retrieve admin document from admins collection
    admin = current_app.config["admin"].find_one({'uname': username})
    print(admin)
    if admin and password == admin["password"]:#and hashpw(password.encode('utf-8'), admin['password']) == admin['password']:
        # Login successful, store admin_id in session
        session['admin_id'] = str(admin['_id'])
        current_app.config["aid"] = str(admin['_id'])
        return redirect("http://localhost:5000/admin/")
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

@auth_bp.route('/login/faculty/', methods=['POST'])
def login_manager():
    data = immutable_to_dict(request.form)
    username = data['uname']
    password = data['password']

    # Retrieve manager document from managers collection
    faculty = current_app.config["faculty"].find_one({'uname': username})

    if faculty and password == faculty["password"]:# and hashpw(password.encode('utf-8'), manager['password']) == manager['password']:
        # Login successful, store manager_id in session
        session['manager_id'] = str(faculty['_id'])
        current_app.config["fid"] = str(faculty['_id'])
        return redirect("http://localhost:5000/faculty/")
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

@auth_bp.route('/logout')
def logout():
    print(session)
    session['user_id'] = None
    session['admin_id'] = None
    session['manager_id'] = None
    current_app.config["aid"] = current_app.config["uid"] = current_app.config["fid"] = None
    return redirect("http://127.0.0.1:5000/")

def login_required(user_type):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check session based on user_type
            if user_type == 'user' and 'user_id' not in session:
                return jsonify({'message': 'Unauthorized access'}), 401
            elif user_type == 'admin' and 'admin_id' not in session:
                return jsonify({'message': 'Unauthorized access'}), 401
            elif user_type == 'manager' and 'manager_id' not in session:
                return jsonify({'message': 'Unauthorized access'}), 401

            # Add user document to request context for further processing
            kwargs['user'] = session.get(f'{user_type}_id')
            return func(*args, **kwargs)

        return wrapper
    return decorator