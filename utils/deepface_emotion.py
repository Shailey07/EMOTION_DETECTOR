"""
Face Emotion Detection - Flask Backend
DeepFace real-time analysis, returns all emotion scores for side panel
Made by: WINTER WOLF 🐺
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import time
import logging
from deepface import DeepFace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# Haar cascade for face detection (fallback if DeepFace region fails)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion colors (BGR -> RGB for frontend)
EMOTION_COLORS = {
    "happy":    (0, 220, 100),    # Greenish
    "sad":      (200, 80, 20),    # Blueish
    "angry":    (0, 0, 220),      # Red
    "fear":     (180, 0, 180),    # Purple
    "surprise": (0, 200, 220),    # Cyan
    "disgust":  (0, 150, 50),     # Dark Green
    "neutral":  (160, 160, 160),  # Gray
}

def analyze_emotions(frame):
    """
    Analyze frame using DeepFace.
    Returns list of dicts: {emotions, dominant_emotion, face_region}
    """
    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        if not isinstance(results, list):
            results = [results]

        output = []
        for res in results:
            emotion_scores = res['emotion']  # dict like {'angry': 1.2, ...}
            dominant = res['dominant_emotion'].lower()
            region = res.get('region', {})
            # If region missing, use fallback face detection
            if not region or region.get('w', 0) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    region = {'x': x, 'y': y, 'w': w, 'h': h}
                else:
                    # fallback values
                    region = {'x': 10, 'y': 10, 'w': 200, 'h': 200}
            output.append({
                'emotions': emotion_scores,
                'dominant': dominant,
                'bbox': [region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)],
                'color': EMOTION_COLORS.get(dominant, (160,160,160))
            })
        return output
    except Exception as e:
        print("DeepFace error:", e)
        return []

# History storage
history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image'})

        # Analyze emotions
        detections = analyze_emotions(frame)

        if not detections:
            return jsonify({'success': False, 'error': 'No face detected'})

        # Prepare response
        results = []
        for det in detections:
            results.append({
                'dominant': det['dominant'],
                'emotions': det['emotions'],   # full dict of all 7 emotions
                'bbox': det['bbox'],
                'color': det['color']
            })
            # Save to history (dominant emotion only)
            from datetime import datetime
            emoji_map = {
                'happy': '😊', 'sad': '😢', 'angry': '😠', 'fear': '😨',
                'surprise': '😲', 'disgust': '🤢', 'neutral': '😐'
            }
            emoji = emoji_map.get(det['dominant'], '😐')
            history.append({
                'timestamp': datetime.now().isoformat(),
                'emotion': f"{emoji} {det['dominant'].capitalize()}",
                'confidence': det['emotions'][det['dominant'].capitalize()] / 100
            })
            if len(history) > 30:
                history.pop(0)

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        print("Detection error:", e)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'success': True, 'history': history})

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    print("\n🐺 WINTER WOLF EMOTION DETECTION (DeepFace) - Web")
    print("📍 Server starting at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)