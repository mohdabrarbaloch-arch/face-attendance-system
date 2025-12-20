import cv2
import torch
from ultralytics import YOLO
try:
    import mediapipe as mp
    MEDIAPIPE_INSTALLED = True
except ImportError:
    MEDIAPIPE_INSTALLED = False
import numpy as np

class VisionEngine:
    def __init__(self):
        # Load YOLOv8 model (Nano version for speed)
        self.model = YOLO('yolov8n.pt') 
        
        # Initialize MediaPipe Face Mesh for drowsiness/mood
        self.mediapipe_available = False
        self.face_mesh = None
        
        if MEDIAPIPE_INSTALLED:
            try:
                # Check for solutions attribute (known issue on Python 3.14)
                if hasattr(mp, 'solutions'):
                    self.mp_face_mesh = mp.solutions.face_mesh
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.mediapipe_available = True
                else:
                    print("⚠️ MediaPipe installed but 'solutions' module is missing (Python 3.14 incompatibility).")
            except Exception as e:
                print(f"⚠️ MediaPipe Initialization Error: {e}. Landmark analysis will be disabled.")
        
        if not self.mediapipe_available:
            # Fallback to OpenCV Haar Cascades for eyes
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Drowsiness detection constants
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 15
        self.counter = 0

    def get_eye_aspect_ratio(self, landmarks, eye_indices):
        # Simplistic EAR calculation
        try:
            # Vertical landmarks
            v1 = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
            v2 = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
            # Horizontal landmark
            h = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
            ear = (v1 + v2) / (2.0 * h)
            return ear
        except:
            return 0

    def process_frame(self, frame):
        # OPTIMIZATION: Resize frame for much faster inference
        small_frame = cv2.resize(frame, (320, 240))
        
        # 1. Object Detection (YOLO) - Fast Inference
        results = self.model(small_frame, verbose=False, imgsz=320)[0]
        detections = []
        counts = {"person": 0, "cell phone": 0, "laptop": 0, "total": 0}
        
        # Scale boxes back to original size
        h, w = frame.shape[:2]
        scale_x, scale_y = w/320, h/240
        
        for box in results.boxes:
            cls = int(box.cls[0])
            name = results.names[cls]
            conf = float(box.conf[0])
            if conf < 0.3: continue
            
            # Count specific objects for Intelligence
            if name in counts:
                counts[name] += 1
            counts["total"] += 1
            
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(coords[0]*scale_x), int(coords[1]*scale_y), int(coords[2]*scale_x), int(coords[3]*scale_y)
            detections.append({'name': name, 'conf': conf, 'box': [x1, y1, x2, y2]})
            
            # Draw box (only for non-person if we want clarity, or all)
            color = (0, 255, 0) if name == "person" else (255, 165, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{name}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2. Mood & Drowsiness
        status = "Normal"
        mood = "Neutral"
        proximity_alert = False

        if self.mediapipe_available and self.face_mesh:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = self.face_mesh.process(rgb_frame)
            
            if mp_results.multi_face_landmarks:
                for face_landmarks in mp_results.multi_face_landmarks:
                    # Basic Mood Estimation (Simplified logic based on landmark proximity)
                    # This is a heuristic placeholder
                    mood = "Neutral/Serious"
                    pass
        else:
            # Fallback Eye Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)
            if len(eyes) > 0:
                status = "Focus: Active"
            else:
                status = "Focus: Eyes not detected"

        return frame, detections, status, counts, mood

if __name__ == "__main__":
    engine = VisionEngine()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        processed_frame, dets, status = engine.process_frame(frame)
        cv2.imshow('Vision Test', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
