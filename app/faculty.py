from flask import Blueprint, render_template, redirect, Response, current_app

faculty_bp = Blueprint('faculty', __name__, url_prefix='/faculty/')

@faculty_bp.route("/")
def home():
    stop_cam()
    db_util = current_app.config['db_util']
    return render_template("faculty/home.html",
                           counts = {
                               "students" : db_util.count("student"),
                               "reports" : db_util.count("report")
                           })

def stop_cam():
    detector = current_app.config['detector']
    if detector.cap and detector.cap.isOpened():
        print("Released !")
        detector.cap.release()

@faculty_bp.route('/stop',methods=['POST'])
def stop():
    stop_cam()
    detector = current_app.config['detector']
    db_util = current_app.config['db_util']
    attendace_data = detector.data
    students = list(db_util.get_all("student"))
    print([i["regNo"] for i in list(students)], attendace_data)
    report = [f"{student['regNo']}-{student['name']}: Present" for student in students if(student['regNo'] in ".".join(attendace_data))]
    report.extend([f"{student['regNo']}-{student['name']}: Absent" for student in students if(student['regNo'] not in ".".join(attendace_data))])
    if(db_util.count("report") == 0):
        new_report_id = 101123
    else:
        new_report_id = db_util.get_last('report')["report_id"] + 1
    report = {"report_id" : new_report_id, "summary" : report}
    db_util.insert("report", report)
    if detector.cap and detector.cap.isOpened():
        detector.cap.release()
    return redirect('reports')

@faculty_bp.route("/reports/")
def reports():
    db_util = current_app.config['db_util']
    return render_template("/faculty/reports.html", reports = db_util.get_all("report"))

@faculty_bp.route("/attendance/")
def take_attendance():
    detector = current_app.config['detector']
    stop_cam()
    detector.data = []
    return render_template("faculty/take_attendance.html")

@faculty_bp.route('/video_capture')
def video_capture():
    detector = current_app.config['detector']
    return Response(detector.capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@faculty_bp.route("/classes/")
def classes():
    stop_cam()
    return render_template("faculty/classes.html")

@faculty_bp.route("/profile")
def about():
    stop_cam()
    return redirect('home')