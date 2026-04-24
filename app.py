"""
Human Emotion Detection System - Complete Backend
Made by: WINTER WOLF 🐺
Supports: Live Camera + Image Upload
"""

from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import json
from datetime import datetime
import traceback

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

# Global camera object
camera = None

# Try to import DeepFace (accurate emotion detection)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully!")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not installed. Install with: pip install deepface")
    print("⚠️ Falling back to OpenCV basic detection")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion mapping with emojis
emotion_emojis = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😲',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐'
}

# Fallback simple detection (when DeepFace not available)
def detect_emotion_simple(face_roi):
    if face_roi is None or face_roi.size == 0:
        return 'neutral', 0.5
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    if avg_brightness > 150:
        return 'happy', 0.6
    elif avg_brightness < 80:
        return 'sad', 0.6
    elif avg_brightness > 130 and avg_brightness < 150:
        return 'surprise', 0.5
    else:
        return 'neutral', 0.7

def detect_emotion_deepface(face_roi):
    """Detect emotion using DeepFace (accurate)"""
    if face_roi is None or face_roi.size == 0:
        return 'neutral', 0.5
    temp_path = 'temp_face.jpg'
    cv2.imwrite(temp_path, face_roi)
    try:
        result = DeepFace.analyze(
            img_path=temp_path,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion] / 100
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return emotion.lower(), confidence
    except Exception as e:
        print(f"DeepFace error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return 'neutral', 0.5

def detect_emotion(face_roi):
    """Main emotion detection function"""
    if DEEPFACE_AVAILABLE:
        return detect_emotion_deepface(face_roi)
    else:
        return detect_emotion_simple(face_roi)

def generate_frames():
    """Live video streaming with emotion detection"""
    global camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ Camera not accessible")
        return
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            emotion, confidence = detect_emotion(face_roi)
            emoji = emotion_emojis.get(emotion, '😐')
            if emotion == 'happy':
                color = (0, 255, 0)
            elif emotion == 'sad':
                color = (255, 0, 0)
            elif emotion == 'angry':
                color = (0, 0, 255)
            elif emotion == 'surprise':
                color = (0, 255, 255)
            else:
                color = (128, 128, 128)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            label = f"{emoji} {emotion.upper()} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            bar_width = int(w * confidence)
            cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), color, -1)
            cv2.rectangle(frame, (x, y+h+5), (x+w, y+h+15), (255, 255, 255), 1)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        # ✅ FIXED LINE (NO SYNTAX ERROR)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and return emotion detection result"""
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
                'confidence': float(confidence),
                'bbox': [int(x), int(y), int(w), int(h)]
            })
            # Save to history (in-memory)
            save_history(emotion, confidence)
        if not results:
            return jsonify({'success': False, 'error': 'No face detected in uploaded image'})
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"Detection error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)})

# In-memory history storage
detection_history = []

def save_history(emotion, confidence):
    detection_history.append({
        'timestamp': datetime.now().isoformat(),
        'emotion': emotion.capitalize(),
        'confidence': confidence
    })
    if len(detection_history) > 50:
        detection_history.pop(0)

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'success': True, 'history': detection_history})

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎭 HUMAN EMOTION DETECTION SYSTEM")
    print("🐺 Created by WINTER WOLF")
    print("="*60)
    if DEEPFACE_AVAILABLE:
        print("✅ DeepFace: ACTIVE (accurate detection)")
    else:
        print("⚠️ DeepFace: NOT INSTALLED (using basic detection)")
        print("💡 For better accuracy, run: pip install deepface")
    print("\n🚀 Server starting...")
    print("📍 Open: http://127.0.0.1:5000")
    print("🔴 Press Ctrl+C to stop\n")
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)