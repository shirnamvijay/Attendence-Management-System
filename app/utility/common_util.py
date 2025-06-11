import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def face_convertor(face):
    x, y, w, h = face_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))[0]
    face = face[y:y + h, x:x + w]
    return Image.fromarray(face).resize((217, 217))

def immutable_to_dict(data):
    resultant_data = {}
    for key in data.to_dict():
        value = data.getlist(key)
        if(len(value) == 1):
            value = value[0]
        resultant_data[key] = value
    return resultant_data