"""
Real Emotion Detection using DeepFace
This actually works!
"""

from deepface import DeepFace
import cv2
import numpy as np

print("🎭 Loading DeepFace Emotion Detector...")
print("⏳ First time will download models (2-3 minutes)")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n✅ Camera ready! DeepFace will detect REAL emotions!")
print("🎮 Press 'q' to quit\n")

emotion_colors = {
    'happy': (0, 255, 0),
    'sad': (255, 0, 0),
    'angry': (0, 0, 255),
    'surprise': (255, 165, 0),
    'fear': (128, 0, 128),
    'disgust': (139, 69, 19),
    'neutral': (128, 128, 128)
}

emotion_emojis = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😲',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐'
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        # Save temp face image
        temp_path = 'temp_face.jpg'
        cv2.imwrite(temp_path, face_roi)
        
        try:
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
            
            # Get dominant emotion
            emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][emotion] / 100
            
            # Get emoji and color
            emoji = emotion_emojis.get(emotion, '😐')
            color = emotion_colors.get(emotion, (0, 255, 0))
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw label
            label = f"{emoji} {emotion.upper()} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Confidence bar
            bar_width = int(w * confidence)
            cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), color, -1)
            cv2.rectangle(frame, (x, y+h+5), (x+w, y+h+15), (255, 255, 255), 1)
            
        except Exception as e:
            cv2.putText(frame, "Processing...", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Info
    cv2.putText(frame, "DeepFace Emotion Detection - Press q", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('REAL Emotion Detection - Press q', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()