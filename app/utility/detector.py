import cv2
from .resnet_util import ResNetRunner
from PIL import Image

class Detector():
    def __init__(self):
        self.runner = ResNetRunner(3)
        # Load Haar Cascade for face detection in the webcam
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Capture video from webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.release()
        self.data = []

    # Define the transformation
    def preprocess_image(self, image, output_shape=(217, 217)):
        return image.resize(output_shape)

    def capture_by_frames(self):
        self.cap.open(0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert frame to grayscale for face detection
            #gray = cv2.cvtColor(frame)
            faces = self.face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                #image = preprocess_image(Image.fromarray(face))
                image = Image.fromarray(face)
                image = self.preprocess_image(image)
                # Invokes ResNet predictor to predict the class of the Detected Face
                output, confidence = self.runner.prediction(image)

                # Get the class with the highest confidence
                #class_id = np.argmax(output)
                #confidence = 0.5#output[0][class_id]
                #print(class_id, runner.classes)
                # Display the prediction on the frame
                label = output # f"{output}: {confidence:.2f}"
                self.data.append(label)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                ret1, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                '''cv2.imshow('Face Recognition', frame)
        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
            cap.release()
            cv2.destroyAllWindows()'''