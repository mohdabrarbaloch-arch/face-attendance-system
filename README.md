# AI Vision Sentinel Pro

Advanced cloud-deployed face recognition attendance system with YOLOv8 object detection, mask compliance monitoring, attention tracking, and PDF report generation.

**Live Demo:** [Streamlit Cloud](#) _(add link after deployment)_

## Architecture

```
User Input (Camera/Upload)
         │
         ▼
┌─────────────────────┐
│   Vision Engine     │  ← YOLOv8 object detection
│   (YOLOv8 + MP)     │  ← MediaPipe face landmarks
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Face Classifier    │  ← Haar cascade + histogram matching
│  (OpenCV)           │  ← Temporal smoothing
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Advanced Detectors  │  ← Mask detection (HSV analysis)
│                     │  ← Attention tracking (eye detection)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Attendance Log    │  ← CSV storage
│   + PDF Reports     │  ← ReportLab generation
└─────────────────────┘
```

## Features

### Core
- **Face Recognition** - Spatial histogram matching with CLAHE preprocessing and temporal smoothing
- **Object Detection** - YOLOv8 nano for real-time person, phone, and laptop detection
- **Attendance Logging** - Automatic CSV-based attendance with duplicate prevention

### Advanced
- **Mask Detection** - HSV color analysis + mouth visibility checking
- **Attention Tracking** - Eye cascade detection with focus scoring
- **Intruder Gallery** - Automatic threat screenshot capture with retention management
- **Face Vault** - Manual registration, gallery-based enrollment, identity revocation
- **PDF Reports** - Attendance reports and security briefs via ReportLab

### UI
- Midnight Onyx glassmorphism theme
- 5-tab command center (Live, Analytics, Gallery, Vault, Logs)
- Real-time metrics dashboard with security scoring
- System health monitoring (CPU, memory, uptime)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Plotly Express |
| Vision | OpenCV, YOLOv8, MediaPipe |
| ML | PyTorch, Ultralytics |
| Reports | ReportLab |
| Data | Pandas, NumPy |
| System | psutil |

## Deployment

### Streamlit Cloud (Recommended)
1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo and set:
   - **Branch:** `main`
   - **Main file:** `streamlit_app.py`
   - **Python version:** 3.11+ (MediaPipe compatible)

### Local
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Project Structure

```
Face_Attendance/
├── streamlit_app.py          # Main entry point
├── vision_engine.py          # YOLOv8 + MediaPipe processing
├── face_classifier.py        # Histogram-based face matching
├── advanced_detectors.py     # Mask + attention detection
├── report_generator.py       # PDF report generation
├── requirements.txt          # Dependencies
├── known_faces/              # Registered face database
├── screenshots/              # Auto-captured threat images
├── data/                     # Runtime data directory
└── attendance_log.csv        # Attendance records
```

## How It Works

1. **Input** - Capture via device camera (`st.camera_input`) or upload image
2. **Detection** - YOLOv8 identifies objects; Haar cascades locate faces
3. **Classification** - 4x4 spatial histograms + MSE scoring matches faces against vault
4. **Analysis** - Mask detection (HSV), attention scoring (eye cascade)
5. **Logging** - Known faces get attendance entry; unknown faces trigger threat capture
6. **Reporting** - Generate PDF attendance or security briefs on demand

## Author

**Abrar Baloch** | [GitHub](https://github.com/mohdabrarbaloch-arch)
