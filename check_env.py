import sys
import os

print(f"Python Version: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
except ImportError:
    print("❌ OpenCV (cv2) NOT FOUND")

try:
    import streamlit
    print(f"✅ Streamlit version: {streamlit.__version__}")
except ImportError:
    print("❌ Streamlit NOT FOUND")

try:
    from ultralytics import YOLO
    print("✅ YOLO (ultralytics) FOUND")
except ImportError:
    print("❌ YOLO (ultralytics) NOT FOUND")

try:
    import mediapipe as mp
    print("✅ Mediapipe FOUND")
except ImportError:
    print("❌ Mediapipe NOT FOUND")

try:
    import pandas as pd
    print(f"✅ Pandas version: {pd.__version__}")
except ImportError:
    print("❌ Pandas NOT FOUND")
