from app import create_ams_app

if __name__ == "__main__":
    app = create_ams_app()
    app.run(debug=True)