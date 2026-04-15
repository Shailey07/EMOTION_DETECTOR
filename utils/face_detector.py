"""
Face Detection Module
Made by: Shailendra Meghwal
"""

import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        """Initialize face detector"""
        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Alternative cascade for better detection
        self.alt_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        
        print("✅ Face Detector Initialized")
    
    def detect_faces(self, frame, method='default'):
        """
        Detect faces in frame
        Returns: List of (x, y, w, h)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improve contrast
        gray = cv2.equalizeHist(gray)
        
        if method == 'default':
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                maxSize=(400, 400)
            )
        else:
            faces = self.alt_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(50, 50)
            )
        
        return faces
    
    def get_face_count(self, frame):
        """Return number of faces detected"""
        return len(self.detect_faces(frame))
    
    def draw_faces(self, frame, faces, color=(0, 255, 0)):
        """Draw rectangles around faces"""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        return frame

# Test function
if __name__ == "__main__":
    print("Testing Face Detector...")
    detector = FaceDetector()
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detector.detect_faces(frame)
        frame = detector.draw_faces(frame, faces)
        
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Detector Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()