import cv2
import os
import numpy as np
import time

class FaceClassifier:
    def __init__(self, known_faces_dir='known_faces'):
        self.known_faces_dir = known_faces_dir
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.known_faces = []      # resized grayscale faces
        self.known_hists = []      # histograms
        self.known_names = []      # Roll IDs

        self.last_reg_time = 0
        self.unknown_counter = {}  # stability tracking
        
        # Stability tracking for recognized names (Temporal Smoothing)
        self.name_history = {} # face_key -> [last_5_names]
        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.load_known_faces()

    # --------------------------------------------------
    def load_known_faces(self):
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)

        self.known_faces.clear()
        self.known_hists.clear()
        self.known_names.clear()

        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(self.known_faces_dir, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Apply CLAHE for consistent lighting
                img = self.clahe.apply(img)
                img = cv2.resize(img, (100, 100))
                self.known_faces.append(img)
                self.known_names.append(os.path.splitext(filename)[0])

                # Spatial Histogram (4x4 Grid)
                grid_size = 4
                h, w = img.shape
                gh, gw = h // grid_size, w // grid_size
                spatial_hists = []
                for r in range(grid_size):
                    for c in range(grid_size):
                        grid_roi = img[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
                        hist = cv2.calcHist([grid_roi], [0], None, [64], [0, 256])
                        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                        spatial_hists.append(hist)
                self.known_hists.append(spatial_hists)

        print(f"Loaded {len(self.known_names)} known faces with spatial grids.")

    # --------------------------------------------------
    def classify_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        face_locations = []
        face_names = []
        current_time = time.time()

        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))

            face_roi = gray[y:y+h, x:x+w]
            
            # Pre-processing (CLAHE)
            face_roi = self.clahe.apply(face_roi)
            face_roi_resized = cv2.resize(face_roi, (100, 100))

            # Matching logic for current frame...
            grid_size = 4
            h, w = face_roi_resized.shape
            gh, gw = h // grid_size, w // grid_size
            target_spatial_hists = []
            for r in range(grid_size):
                for c in range(grid_size):
                    grid_roi = face_roi_resized[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
                    hist = cv2.calcHist([grid_roi], [0], None, [64], [0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    target_spatial_hists.append(hist)

            best_match = "Unknown"
            max_similarity = -1
            matched_name = None

            for i, (known_img, known_spatial_hists) in enumerate(
                zip(self.known_faces, self.known_hists)
            ):
                # Compare each grid cell
                grid_scores = []
                for thist, khist in zip(target_spatial_hists, known_spatial_hists):
                    grid_scores.append(cv2.compareHist(thist, khist, cv2.HISTCMP_CORREL))
                
                hist_score = np.mean(grid_scores)

                err = np.mean(
                    (face_roi_resized.astype("float") - known_img.astype("float")) ** 2
                )
                mse_score = max(0, 1 - (err / 4000)) # Strict but fair

                combined_score = (hist_score * 0.7) + (mse_score * 0.3)

                if combined_score > max_similarity:
                    max_similarity = combined_score
                    matched_name = self.known_names[i]

            # Final threshold decision
            if max_similarity >= 0.70: # Threshold increased to 70% for better accuracy
                best_match = matched_name
            else:
                best_match = "Unknown"

            # -------- TEMPORAL SMOOTHING (Voting) --------
            # Use spatial grid for identity persistence
            grid_x, grid_y = x // 80, y // 80 
            pers_key = f"{grid_x}_{grid_y}"
            
            if pers_key not in self.name_history:
                self.name_history[pers_key] = []
            
            self.name_history[pers_key].append(best_match)
            if len(self.name_history[pers_key]) > 6: # Keep last 6 frames
                self.name_history[pers_key].pop(0)
            
            # Get the most frequent name in history
            from collections import Counter
            counts = Counter(self.name_history[pers_key])
            best_match = counts.most_common(1)[0][0]

            # Auto-registration disabled per user request. 
            # Enrollments must now be done manually through the app UI.

            face_names.append(best_match)

        return face_locations, face_names

    def register_face(self, face_roi, name):
        """Manually register a face ROI with a specific name."""
        if face_roi is None or not name:
            return False
        
        # Prevent overwriting if user didn't explicitly ask
        save_path = os.path.join(self.known_faces_dir, f"{name}.jpg")
        if os.path.exists(save_path):
            print(f"⚠️ Warning: Name {name} already exists. Overwriting...")
            
        cv2.imwrite(save_path, face_roi)
        self.load_known_faces()
        print(f"Manually registered: {name}")
        return True


# --------------------------------------------------
if __name__ == "__main__":
    clf = FaceClassifier()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        locs, names = clf.classify_face(frame)
        for (top, right, bottom, left), name in zip(locs, names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Classifier Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
 