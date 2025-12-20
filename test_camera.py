import cv2
import sys

def test_camera(index, backend=None):
    if backend is not None:
        print(f"Testing Index {index} | Backend {backend}...", end=" ")
        cap = cv2.VideoCapture(index, backend)
    else:
        print(f"Testing Index {index} | Backend AUTO...", end=" ")
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print("❌ FAILED (Could not open)")
        return False
    
    ret, frame = cap.read()
    if ret:
        print(f"✅ SUCCESS! Frame: {frame.shape}")
        cap.release()
        return True
    else:
        print("❌ FAILED (Opened but no frame)")
        cap.release()
        return False

print("\n--- CAMERA DIAGNOSTICS START ---")
print(f"Python: {sys.executable}")
print(f"OpenCV: {cv2.__version__}")

backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, None]
backend_names = ["DSHOW", "MSMF", "VFW", "AUTO"]

found = False
for i in range(5): # Test up to 5 indices
    for b_idx, b in enumerate(backends):
        if test_camera(i, b):
            found = True

if not found:
    print("\n⚠️ No working camera found. Please ensure:")
    print("1. Camera is plugged in.")
    print("2. No other app (Zoom, Teams, another Streamlit) is using it.")
    print("3. Try restarting your computer if errors persist.")

print("--- CAMERA DIAGNOSTICS END ---\n")
