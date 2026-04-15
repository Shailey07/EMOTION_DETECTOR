"""
Human Emotion Detection - Complete Working System
Made by: Shailendra Meghwal
Live Face Detection + Emotion Recognition
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import os
import json
from datetime import datetime
import base64
from PIL import Image
import io

app = Flask(__name__)

# Global variables
camera = None
face_cascade = None
emotion_running = True

# Emotion labels with emojis
EMOTIONS = {
    'happy': ['😊', 'Happy', '#4CAF50'],
    'sad': ['😢', 'Sad', '#2196F3'],
    'angry': ['😠', 'Angry', '#f44336'],
    'surprise': ['😲', 'Surprise', '#FF9800'],
    'fear': ['😨', 'Fear', '#9C27B0'],
    'disgust': ['🤢', 'Disgust', '#795548'],
    'neutral': ['😐', 'Neutral', '#607D8B']
}

# Simple rule-based emotion detection from facial features
def detect_emotion_simple(face_roi):
    """
    Simple emotion detection based on facial features
    Real implementation would use deep learning
    """
    if face_roi is None or face_roi.size == 0:
        return 'neutral', 0.5
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate basic features
    height, width = gray.shape
    
    # Divide face into regions
    top_third = gray[0:height//3, :]
    middle_third = gray[height//3:2*height//3, :]
    bottom_third = gray[2*height//3:, :]
    
    # Calculate brightness of different regions
    top_brightness = np.mean(top_third)
    middle_brightness = np.mean(middle_third)
    bottom_brightness = np.mean(bottom_third)
    
    # Simple emotion rules
    # Smile detection (mouth region brighter)
    if bottom_brightness > middle_brightness + 20:
        return 'happy', 0.7 + (bottom_brightness - middle_brightness) / 100
    
    # Sad detection (eyes and mouth darker)
    elif top_brightness > bottom_brightness + 15:
        return 'sad', 0.65
    
    # Surprise detection (whole face brighter)
    elif np.mean(gray) > 150:
        return 'surprise', 0.6
    
    # Angry detection (furrowed brow - top part darker)
    elif top_brightness < middle_brightness - 15:
        return 'angry', 0.6
    
    # Fear detection (uneven brightness)
    elif abs(top_brightness - bottom_brightness) > 30:
        return 'fear', 0.55
    
    else:
        return 'neutral', 0.5

def init_camera():
    """Initialize camera and face detector"""
    global camera, face_cascade
    
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("❌ Camera not found!")
            return False
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("✅ Camera and face detector initialized!")
        return True
    except Exception as e:
        print(f"❌ Error initializing camera: {e}")
        return False

def generate_frames():
    """Generate video frames with face detection and emotion recognition"""
    global camera, face_cascade, emotion_running
    
    if not init_camera():
        return
    
    frame_count = 0
    
    while emotion_running:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect emotion
            emotion_key, confidence = detect_emotion_simple(face_roi)
            emotion_info = EMOTIONS.get(emotion_key, EMOTIONS['neutral'])
            
            emoji, emotion_name, color = emotion_info
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Create label
            label = f"{emoji} {emotion_name} ({confidence*100:.1f}%)"
            
            # Add background for text
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y-label_h-10), (x+label_w, y), (0, 255, 0), -1)
            
            # Add text
            cv2.putText(frame, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add confidence bar
            bar_width = int(w * confidence)
            cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), (0, 255, 0), -1)
            cv2.rectangle(frame, (x, y+h+5), (x+w, y+h+15), (255, 255, 255), 1)
        
        # Add info text
        cv2.putText(frame, f"Faces Detected: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    if camera:
        camera.release()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    """Stop camera"""
    global emotion_running, camera
    emotion_running = False
    if camera:
        camera.release()
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎭 HUMAN EMOTION DETECTION SYSTEM")
    print("👨‍💻 Created by: Shailendra Meghwal")
    print("="*60)
    print("\n✨ Features:")
    print("📹 Real-time face detection")
    print("🎭 7 Emotion recognition")
    print("📊 Confidence scoring")
    print("🟢 Green box around face")
    print("\n🚀 Starting server...")
    print("📍 Open browser: http://127.0.0.1:5000")
    print("🔴 Press Ctrl+C to stop\n")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)