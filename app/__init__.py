from datetime import timedelta

from flask import Flask, current_app

from app.admin import admin_bp
from app.student import student_bp
from app.faculty import faculty_bp
from app.base import base_bp
from app.auth import auth_bp
from app.utility.detector import Detector
from app.utility.db_util import DBUtil
import os
from datetime import datetime

def create_ams_app():

    app = Flask(__name__)
    app.secret_key = 'ams_app'
    app.permanent_session_lifetime = timedelta(minutes=15)
    app.template_folder = "../templates"
    app.static_folder = "../static"
    db_util = DBUtil()
    detector = Detector()
    detector.runner.num_classes = db_util.count("student")
    detector.runner.classes = [f'{s["regNo"]}-{s["name"]}' for s in db_util.get_all("student")]
    app.config["detector"] = detector
    app.config["db_util"] = db_util
    app.register_blueprint(base_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(faculty_bp)
    app.register_blueprint(student_bp)
    app.config["users"] = db_util.db['student']
    app.config["admin"] = db_util.db['admin']
    app.config["faculty"] = db_util.db['faculty']
    return app