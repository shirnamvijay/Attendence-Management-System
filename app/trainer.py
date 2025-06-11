from app.utility.resnet_util import ResNetRunner
from app.utility import dummy
from pathlib import Path
import os

if(__name__ == "__main__"):
    print("Current Path :", Path.cwd())
    #print(os.path.dirname(__file__))
    runner = ResNetRunner(3)
    #runner.train()
    runner.test_prediction()
    #dummy.disp()