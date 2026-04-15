"""
Enhanced Emotion Detection - With Angry Boost
Made by: Shailendra Meghwal
"""

from deepface import DeepFace
import cv2
import numpy as np
from collections import deque

class EnhancedEmotionDetector:
    def __init__(self):
        # Emotion mapping
        self.emotion_colors = {
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'surprise': (255, 165, 0),
            'fear': (128, 0, 128),
            'disgust': (139, 69, 19),
            'neutral': (128, 128, 128)
        }
        
        self.emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😲',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐'
        }
        
        # For smoothing predictions
        self.emotion_history = deque(maxlen=5)
        
        # Angry detection thresholds
        self.angry_features = {
            'brow_furrow': True,    # Eyebrows down and together
            'upper_eyelid': False,   # Upper eyelid is raised
            'lower_eyelid': True,    # Lower eyelid is tense
            'lip_corners': False,    # Lip corners pulled down or straight
            'jaw_clench': True       # Jaw clenching
        }
        
        print("✅ Enhanced Emotion Detector Ready!")
        print("🔥 Angry detection boosted!")
    
    def detect_emotion(self, face_img):
        """Detect emotion with angry boost"""
        temp_path = 'temp_face.jpg'
        cv2.imwrite(temp_path, face_img)
        
        try:
            # DeepFace analysis
            result = DeepFace.analyze(
                img_path=temp_path, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][emotion] / 100
            
            # BOOST FOR ANGRY - Additional detection
            angry_boost = self._detect_angry_boost(face_img)
            
            if angry_boost > 0.6:  # If angry features detected
                # Check if angry score is higher than current emotion
                angry_conf = result[0]['emotion']['angry'] / 100
                
                if angry_conf > 0.4 or angry_boost > 0.5:
                    emotion = 'angry'
                    confidence = max(angry_conf, angry_boost)
                    print(f"🔥 ANGRY BOOST ACTIVATED! (Boost: {angry_boost:.2f})")
            
            return emotion, confidence
            
        except Exception as e:
            return 'neutral', 0.5
    
    def _detect_angry_boost(self, face_img):
        """Additional angry detection using facial features"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Eyebrow position (furrowed brows = angry)
        eyebrow_region = gray[0:h//4, :]
        eyebrow_intensity = np.mean(eyebrow_region)
        
        # 2. Mouth tension (clenched jaw)
        mouth_region = gray[3*h//4:h, :]
        mouth_intensity = np.mean(mouth_region)
        
        # 3. Overall face tension
        face_tension = np.std(gray)
        
        # 4. Symmetry (angry faces are less symmetric)
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        symmetry = 1 - np.mean(np.abs(left_half - right_half)) / 255
        
        # Calculate angry score
        angry_score = 0
        
        # Furrowed brows = lower eyebrow brightness
        if eyebrow_intensity < 100:
            angry_score += 0.3
        
        # Mouth tension
        if mouth_intensity < 80:
            angry_score += 0.3
        
        # High face tension
        if face_tension > 60:
            angry_score += 0.2
        
        # Low symmetry
        if symmetry < 0.7:
            angry_score += 0.2
        
        return min(angry_score, 0.95)
    
    def draw_emotion(self, frame, face, emotion, confidence):
        x, y, w, h = face
        emoji = self.emotion_emojis.get(emotion, '😐')
        color = self.emotion_colors.get(emotion, (0, 255, 0))
        
        # Special angry border
        if emotion == 'angry':
            thickness = 4
            # Add extra red glow
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 255), 2)
        else:
            thickness = 3
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Label with emoji
        label = f"{emoji} {emotion.upper()} ({confidence*100:.1f}%)"
        
        # Background for text
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y-label_h-10), (x+label_w, y), color, -1)
        
        # Text
        cv2.putText(frame, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Confidence bar
        bar_width = int(w * confidence)
        cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), color, -1)
        cv2.rectangle(frame, (x, y+h+5), (x+w, y+h+15), (255, 255, 255), 1)
        
        return frame

def main():
    print("\n" + "="*60)
    print("🎭 ENHANCED EMOTION DETECTION")
    print("👨‍💻 Made by: Shailendra Meghwal")
    print("🔥 Angry detection boosted!")
    print("="*60 + "\n")
    
    detector = EnhancedEmotionDetector()
    
    # Face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera ready!")
    print("🎮 Press 'q' to quit")
    print("🔥 Try making an angry face - detection boosted!\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        # Process each face
        for face in faces:
            x, y, w, h = face
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect emotion with angry boost
            emotion, confidence = detector.detect_emotion(face_roi)
            
            # Draw on frame
            frame = detector.draw_emotion(frame, face, emotion, confidence)
        
        # Info
        cv2.putText(frame, f"Faces: {len(faces)} | ANGRY BOOST: ON", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Angry face tips
        if len(faces) > 0:
            cv2.putText(frame, "🔥 Make angry face: Furrow brows + Clench jaw", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow('Enhanced Emotion Detection - Press q', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Program ended!")

if __name__ == "__main__":
    main()