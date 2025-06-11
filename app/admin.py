import numpy as np
from flask import Blueprint, render_template, redirect, request, Response, make_response, url_for, current_app
import json
from app.utility.image_util import create_image_from_bytes, base64_to_image, show_image, save_image
from app.utility.common_util import immutable_to_dict, face_convertor
import pathlib
import os
from app.auth import login_required

admin_bp = Blueprint('admin', __name__, url_prefix='/admin/')

@admin_bp.route("/")
def home():
    db_util = current_app.config['db_util']
    return render_template("admin/home.html", counts = {
        "students" : db_util.count("student"),
        "faculty" : db_util.count("faculty"),
        "departments" : db_util.count("department"),
        "classes" : db_util.count("class"),
        "reports" : db_util.count("report")
    })

@admin_bp.route("/add_department/", methods = ["GET", "POST"])
def add_department():
    db_util = current_app.config['db_util']
    if(request.method == "POST"):
        dept_data = immutable_to_dict(request.form)
        if(db_util.count("department", {"name" : dept_data["name"]}) == 0):
            db_util.insert("department", dept_data)
            print("Data Inserted Successfully !")
            return redirect(url_for('admin.view_departments'))
    return render_template("admin/add_department.html")

@admin_bp.route("/view_departments/")
def view_departments():
    db_util = current_app.config['db_util']
    return render_template("admin/view_departments.html", data = db_util.get_all("department"))

@admin_bp.route("/add_faculty/", methods = ["GET", "POST"])
def add_faculty():
    db_util = current_app.config['db_util']
    if(request.method == "POST"):
        fac_data = immutable_to_dict(request.form)
        print(fac_data)
        if(db_util.count("faculty", {"regNo" : fac_data["regNo"]}) == 0):
            fac_data["uname"] = f"{fac_data['regNo']}@ams.com"
            fac_data["password"] = f"{fac_data['name'][:3]}@{fac_data['regNo']}$@"
            db_util.insert("faculty", fac_data)
            print("Data Inserted Successfully !")
            return redirect(url_for('admin.view_faculty'))
    return render_template("admin/add_faculty.html",
                           departments = db_util.get_all("department"),
                           uid = int(db_util.get_last('faculty')["regNo"]) + 1)

@admin_bp.route("/view_faculty/")
def view_faculty():
    db_util = current_app.config['db_util']
    return render_template("admin/view_faculty.html", data = db_util.get_all("faculty"))
@admin_bp.route("/add_class/")
def add_class():
    return render_template("admin/add_class.html")
@admin_bp.route('/start',methods=['POST'])
def start():
    return render_template('admin/detector.html')

@admin_bp.route("/add_student/")
def add_student():
    db_util = current_app.config['db_util']
    return render_template("admin/add_student.html",
                           data = db_util.get_all('department'),
                           uid = int(db_util.get_last('student')["regNo"]) + 1)

@admin_bp.route("/register_student/", methods=["POST"])
def register_student():
    detector = current_app.config['detector']
    db_util = current_app.config['db_util']
    if(request.method == "POST"):
        data = immutable_to_dict(request.form)
        data["images"] = data.pop("images[]")
        data["uname"] = f"{data['regNo']}@ams.com"
        data["password"] = f"{data['name'][:3]}@{data['regNo']}$@"
        with open("dummy.json", "w") as writer:
            json.dump(data, writer)
        for img, cnt in zip(data["images"], range(len(data["images"]))):
            image = create_image_from_bytes(base64_to_image(img))
            save_image(face_convertor(np.array(image)), os.path.join(pathlib.Path().resolve().absolute(), "temp", data["regNo"]), f"{data['regNo']}_img_{cnt}.jpg")
        if(db_util.count("student", {"regNo" : data["regNo"]}) == 0):
            db_util.insert("student", data)
            detector.runner.num_classes = db_util.count("student", {})
            detector.runner.classes = [s["regNo"] for s in db_util.get_all("student")]
            detector.runner.load_data()
            detector.runner.train()
        else:
            print("Data Already Exists !")
        return make_response("The Data Received Successfully !", 200)
    else:
        return make_response("Not Received !", 501)

@admin_bp.route('/video_capture')
def video_capture():
    detector = current_app.config['detector']
    return Response(detector.capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@admin_bp.route("/about/")
def about():
    return redirect('home')

@admin_bp.route("/reports/")
def reports():
    return render_template("admin/reports.html")