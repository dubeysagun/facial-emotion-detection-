from flask import Flask, request, jsonify
import cv2
from keras.models import model_from_json
import numpy as np

app = Flask(__name__)

# Load model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        file.save("temp.jpg")
        image = cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(image, 1.3, 5)
        
        if len(faces) == 0:
            return jsonify({"error": "No face detected"})
        
        for (p, q, r, s) in faces:
            face = image[q:q + s, p:p + r]
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            return jsonify({"emotion": prediction_label})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
