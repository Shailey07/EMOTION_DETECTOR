"""
Human Emotion Detection System - Lightweight for Render
Made by: WINTER WOLF 🐺
No DeepFace, only OpenCV + rule-based detection (stable)
"""

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_emojis = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😲',
    'neutral': '😐'
}

def detect_emotion_simple(face_roi):
    """Rule-based emotion detection (fast, no dependencies)"""
    if face_roi is None or face_roi.size == 0:
        return 'neutral', 0.5
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # Sobel edges for mouth region
    mouth_region = gray[2*len(gray)//3:, :]
    mouth_edges = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)
    edge_density = np.sum(np.abs(mouth_edges)) / mouth_region.size
    
    # Brightness difference between top and bottom
    h, w = gray.shape
    top = gray[:h//2, :].mean()
    bottom = gray[h//2:, :].mean()
    
    if bottom > top + 15:
        return 'happy', 0.7
    elif top > bottom + 20:
        return 'sad', 0.6
    elif edge_density > 30:
        return 'surprise', 0.6
    elif gray.mean() < 80:
        return 'angry', 0.6
    else:
        return 'neutral', 0.7

def detect_emotion(face_roi):
    return detect_emotion_simple(face_roi)

# In-memory history
detection_history = []

def save_history(emotion, confidence):
    detection_history.append({
        'timestamp': datetime.now().isoformat(),
        'emotion': emotion.capitalize(),
        'confidence': confidence
    })
    if len(detection_history) > 50:
        detection_history.pop(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image'})
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            emotion, confidence = detect_emotion(face_roi)
            emoji = emotion_emojis.get(emotion, '😐')
            results.append({
                'emotion': f"{emoji} {emotion.capitalize()}",
                'base_emotion': emotion,
                'confidence': confidence,
                'bbox': [int(x), int(y), int(w), int(h)]
            })
            save_history(emotion, confidence)
        if not results:
            return jsonify({'success': False, 'error': 'No face detected. Adjust lighting or position.'})
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'success': True, 'history': detection_history})

if __name__ == '__main__':
    print("🐺 WINTER WOLF Emotion Detector - Lightweight Mode")
    app.run(host='0.0.0.0', port=5000)