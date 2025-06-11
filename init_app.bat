:: Create virtual environment
python -m venv venv

:: Activate virtual environment
:: ./venv/Scripts/activate

:: Install the dependencies that are listed in ams_app_req.txt
"./venv/Scripts/pip.exe" install -r ams_app_req.txt

:: Display python version
"./venv/Scripts/python.exe" --version

:: Display Log message
echo 'Run start_app.bat file to start app'