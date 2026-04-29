"""
Image Preprocessing Module
"""

import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        print("✅ Image Preprocessor Initialized")
    
    def preprocess_face(self, face_img, target_size=(48, 48)):
        """Preprocess face for emotion detection"""
        if face_img is None or face_img.size == 0:
            return None
        
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Resize
        resized = cv2.resize(gray, target_size)
        
        # Normalize pixel values (0-1)
        normalized = resized.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        processed = np.expand_dims(normalized, axis=-1)
        processed = np.expand_dims(processed, axis=0)
        
        return processed
    
    def enhance_face(self, face_img):
        """Enhance face image quality"""
        if face_img is None:
            return None
        
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Apply histogram equalization
        enhanced = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced

# Test function
if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    print("✅ Preprocessor ready!")