import cv2
import numpy as np
from scipy.spatial import distance as dist

class MaskDetector:
    """Simple mask detection using face landmarks and color analysis"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.mouth_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
    
    def detect_mask(self, frame, face_location):
        """
        Detect if a face has a mask or not
        Returns: (has_mask: bool, confidence: float)
        """
        top, right, bottom, left = face_location
        
        # Extract face ROI
        face_roi = frame[top:bottom, left:right]
        
        if face_roi.size == 0:
            return False, 0.0
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Check for mouth/smile detection (if mouth visible = no mask)
        mouths = self.mouth_cascade.detectMultiScale(
            gray_face, scaleFactor=1.1, minNeighbors=20, minSize=(15, 15)
        )
        
        # If mouth is clearly visible, no mask
        if len(mouths) > 0:
            return False, 0.8
        
        # Analyze lower half of face for mask indicators
        h, w = face_roi.shape[:2]
        lower_face = face_roi[int(h*0.5):, :]
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
        
        # Check for typical mask colors (blue, white, black surgical masks)
        # Blue mask range
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # White mask range
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Black mask range
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(blue_mask, white_mask)
        combined_mask = cv2.bitwise_or(combined_mask, black_mask)
        
        # Calculate percentage of mask-colored pixels
        mask_pixel_ratio = np.count_nonzero(combined_mask) / combined_mask.size
        
        # Decision logic
        if mask_pixel_ratio > 0.3:  # More than 30% mask-colored pixels
            return True, min(0.95, mask_pixel_ratio + 0.3)
        else:
            return False, max(0.6, 1.0 - mask_pixel_ratio)


class AttentionDetector:
    """Detect if person is paying attention to camera"""
    
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 3
    
    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio"""
        # Compute euclidean distances between vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Compute euclidean distance between horizontal eye landmarks
        C = dist.euclidean(eye[0], eye[3])
        # Compute eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_attention(self, frame, face_location):
        """
        Detect if person is paying attention
        Returns: (is_attentive: bool, attention_score: float, status: str)
        """
        top, right, bottom, left = face_location
        
        # Extract face ROI
        face_roi = frame[top:bottom, left:right]
        
        if face_roi.size == 0:
            return False, 0.0, "Unknown"
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray_face, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20)
        )
        
        # No eyes detected = looking away or eyes closed
        if len(eyes) == 0:
            return False, 0.2, "Distracted"
        
        # Both eyes detected = likely looking at camera
        if len(eyes) >= 2:
            # Calculate face center
            face_h, face_w = face_roi.shape[:2]
            face_center_x = face_w // 2
            
            # Check if eyes are roughly centered (person facing camera)
            eyes_centered = 0
            for (ex, ey, ew, eh) in eyes:
                eye_center_x = ex + ew // 2
                if abs(eye_center_x - face_center_x) < face_w * 0.3:
                    eyes_centered += 1
            
            if eyes_centered >= 2:
                return True, 0.85, "Focused"
            elif eyes_centered == 1:
                return True, 0.65, "Partially Focused"
            else:
                return False, 0.4, "Looking Away"
        
        # One eye detected = partially attentive
        elif len(eyes) == 1:
            return True, 0.5, "Partially Focused"
        
        return False, 0.3, "Distracted"
