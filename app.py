"""
Face Emotion Detection - Web Version with Fallback
Made by: WINTER WOLF
"""

import os
import time
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

# Haar cascade for face detection (fallback)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EMOTION_COLORS = {
    "happy": (0, 220, 100),
    "sad": (200, 80, 20),
    "angry": (0, 0, 220),
    "fear": (180, 0, 180),
    "surprise": (0, 200, 220),
    "disgust": (0, 150, 50),
    "neutral": (160, 160, 160),
}

history = []

def detect_faces(frame):
    """OpenCV face detection (fallback if DeepFace fails)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return faces

def analyze_emotions(frame):
    """DeepFace analysis with fallback face detection"""
    try:
        # Try DeepFace first
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        if not isinstance(results, list):
            results = [results]
        
        output = []
        for res in results:
            # Check if DeepFace found face
            region = res.get('region', {})
            if region.get('w', 0) == 0:
                # DeepFace didn't detect face, use OpenCV
                faces = detect_faces(frame)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    region = {'x': x, 'y': y, 'w': w, 'h': h}
                else:
                    region = {'x': 50, 'y': 50, 'w': 200, 'h': 200}
            
            x = region.get('x', 50)
            y = region.get('y', 50)
            w = region.get('w', 200)
            h = region.get('h', 200)
            
            output.append({
                'dominant': res['dominant_emotion'].lower(),
                'emotions': res['emotion'],
                'bbox': [int(x), int(y), int(w), int(h)],
                'color': EMOTION_COLORS.get(res['dominant_emotion'].lower(), (160,160,160))
            })
        return output
    
    except Exception as e:
        print(f"DeepFace error: {e}")
        # Fallback: use OpenCV face detection only
        faces = detect_faces(frame)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return [{
                'dominant': 'neutral',
                'emotions': {'happy': 0, 'sad': 0, 'angry': 0, 'fear': 0, 'surprise': 0, 'disgust': 0, 'neutral': 100},
                'bbox': [int(x), int(y), int(w), int(h)],
                'color': (160,160,160)
            }]
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Invalid image'})
        
        results = analyze_emotions(frame)
        
        if not results:
            return jsonify({'success': False, 'error': 'No face detected'})
        
        # Save history
        for r in results:
            dom = r['dominant']
            conf = r['emotions'].get(dom.capitalize(), 50) / 100
            emoji_map = {'happy':'😊','sad':'😢','angry':'😠','fear':'😨','surprise':'😲','disgust':'🤢','neutral':'😐'}
            emoji = emoji_map.get(dom, '😐')
            history.append({
                'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
                'emotion': f"{emoji} {dom.capitalize()}",
                'confidence': conf
            })
            if len(history) > 30:
                history.pop(0)
        
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'success': True, 'history': history})

if __name__ == '__main__':
    print("\n🐺 WINTER WOLF EMOTION DETECTOR")
    print("📍 http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)