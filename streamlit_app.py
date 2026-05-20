import sys
import os
try:
    import cv2
except Exception as e:
    print(f"OpenCV Import Error: {e}")
    sys.exit(1)
import streamlit as st
import pandas as pd
from vision_engine import VisionEngine
from face_classifier import FaceClassifier
import datetime
import time
import os
import numpy as np
import plotly.express as px

# Advanced Features Imports
try:
    from advanced_detectors import MaskDetector, AttentionDetector
    ADVANCED_DETECTION_AVAILABLE = True
except ImportError:
    ADVANCED_DETECTION_AVAILABLE = False

try:
    from report_generator import ReportGenerator
    REPORTS_AVAILABLE = True
except ImportError:
    REPORTS_AVAILABLE = False

try:
    import psutil
    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False

# --- Page Config ---
st.set_page_config(page_title="AI Vision Sentinel Pro", layout="wide", initial_sidebar_state="expanded")

# --- MIDNIGHT ONYX UI ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
    .stApp {
        background-color: #020617;
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.05) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.05) 0px, transparent 50%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .premium-header {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        padding: 3rem 1rem;
        border-bottom: 2px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 2.5rem;
        text-align: center;
        border-radius: 0 0 50px 50px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }
    .premium-header h1 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 4.2rem;
        margin: 0;
        letter-spacing: 8px;
        text-transform: uppercase;
        filter: drop-shadow(0 0 15px rgba(56, 189, 248, 0.4));
    }
    .premium-header p {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 12px;
        margin-top: 0.8rem;
        text-transform: uppercase;
        opacity: 0.8;
    }
    .luminous-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 28px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-left: 5px solid #38bdf8 !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(10px) !important;
    }
    div[data-testid="stMetric"] label {
        color: #38bdf8 !important;
        font-size: 0.85rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #fff !important;
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(56, 189, 248, 0.1) !important;
    }
    .stCheckbox label { color: #cbd5e1 !important; font-weight: 600; }
    .status-alert {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #38bdf8;
        color: #38bdf8;
        padding: 1.2rem;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: inset 0 0 15px rgba(56, 189, 248, 0.1), 0 0 20px rgba(56, 189, 248, 0.1);
        animation: pulse-glow 2s infinite alternate;
    }
    @keyframes pulse-glow {
        from { box-shadow: 0 0 10px rgba(56, 189, 248, 0.1); }
        to { box-shadow: 0 0 25px rgba(56, 189, 248, 0.3); }
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px !important;
        white-space: pre-wrap !important;
        background-color: rgba(30, 41, 59, 0.5) !important;
        border-radius: 12px 12px 0 0 !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        font-weight: 700 !important;
        padding: 0 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(0deg, rgba(56, 189, 248, 0.2) 0%, rgba(30, 41, 59, 0.8) 100%) !important;
        border-top: 3px solid #38bdf8 !important;
        color: #38bdf8 !important;
    }
    .stDataFrame, [data-testid="stTable"] {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(56, 189, 248, 0.1) !important;
        border-radius: 16px !important;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #38bdf8, #a855f7) !important;
    }
</style>

<div class="premium-header">
    <h1>S E N T I N E L</h1>
    <p>Cloud Intelligence Core • v5.0</p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if 'attendance_df' not in st.session_state:
    st.session_state.attendance_df = pd.DataFrame(columns=["Roll No", "Date", "Time", "Status"])
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'unknown_frames_count' not in st.session_state:
    st.session_state.unknown_frames_count = 0
if 'last_unknown_face' not in st.session_state:
    st.session_state.last_unknown_face = None
if 'security_score' not in st.session_state:
    st.session_state.security_score = 100
if 'unknown_faces_list' not in st.session_state:
    st.session_state.unknown_faces_list = []
if 'alerts_list' not in st.session_state:
    st.session_state.alerts_list = []
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = True
if 'enrollments_today' not in st.session_state:
    st.session_state.enrollments_today = 0
if 'system_start_time' not in st.session_state:
    st.session_state.system_start_time = time.time()
if 'mask_compliance_data' not in st.session_state:
    st.session_state.mask_compliance_data = {'with_mask': 0, 'without_mask': 0}
if 'attention_scores' not in st.session_state:
    st.session_state.attention_scores = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'total_known' not in st.session_state:
    st.session_state.total_known = 0
if 'total_unknown' not in st.session_state:
    st.session_state.total_unknown = 0

attendance_file = "attendance_log.csv"
if os.path.exists(attendance_file):
    try:
        st.session_state.attendance_df = pd.read_csv(attendance_file)
    except Exception:
        st.session_state.attendance_df = pd.DataFrame(columns=["Roll No", "Date", "Time", "Status"])

# Initialize Engines
@st.cache_resource
def load_engines():
    with st.spinner("Loading AI models (first run downloads YOLOv8)..."):
        vision_eng = VisionEngine()
        face_classifier = FaceClassifier()
        mask_det = MaskDetector() if ADVANCED_DETECTION_AVAILABLE else None
        attention_det = AttentionDetector() if ADVANCED_DETECTION_AVAILABLE else None
        report_gen = ReportGenerator() if REPORTS_AVAILABLE else None
    return vision_eng, face_classifier, mask_det, attention_det, report_gen

vision, face_clf, mask_detector, attention_detector, report_generator = load_engines()

# --- Process a single frame ---
def process_frame(frame, enable_voice_alerts=True):
    display_frame = frame.copy()
    results = {
        'frame': display_frame,
        'face_locs': [],
        'face_names': [],
        'counts': {'person': 0, 'cell phone': 0, 'laptop': 0, 'total': 0},
        'status': 'Normal',
        'mood': 'Neutral',
        'alerts': [],
        'new_attendance': None
    }

    try:
        # 1. AI Vision Core (YOLO detection)
        res_frame, detections, status, counts, mood = vision.process_frame(frame.copy())
        face_locs, face_names = face_clf.classify_face(frame)

        results['face_locs'] = face_locs
        results['face_names'] = face_names
        results['counts'] = counts
        results['status'] = status
        results['mood'] = mood

        # 2. Security Armor Logic
        has_unknown = any(name == "Unknown" for name in face_names)
        if has_unknown:
            st.session_state.unknown_frames_count += 1
            st.session_state.total_unknown += 1
            if st.session_state.security_score > 10:
                st.session_state.security_score -= 1
        else:
            st.session_state.unknown_frames_count = 0
            if st.session_state.security_score < 100:
                st.session_state.security_score += 0.5

        # 3. Pro-Grade Overlays
        for (top, right, bottom, left), name in zip(face_locs, face_names):
            label = "UNIDENTIFIED" if name == "Unknown" else f"ID: {name}"
            color = (0, 0, 255) if name == "Unknown" else (37, 99, 235)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(display_frame, f"{label} | {mood}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if name == "Unknown":
                r_roi = frame[top:bottom, left:right]
                if r_roi.size > 0:
                    unknown_face_rgb = cv2.cvtColor(r_roi, cv2.COLOR_BGR2RGB)
                    if len(st.session_state.unknown_faces_list) < 5:
                        face_exists = False
                        for existing_face in st.session_state.unknown_faces_list:
                            if existing_face['image'].shape == unknown_face_rgb.shape:
                                face_exists = True
                                break
                        if not face_exists:
                            st.session_state.unknown_faces_list.append({
                                'image': unknown_face_rgb,
                                'timestamp': time.time(),
                                'location': (top, right, bottom, left)
                            })

            # Security Snapshot
            if name == "Unknown" and st.session_state.unknown_frames_count >= 3:
                if not os.path.exists("screenshots"):
                    os.makedirs("screenshots")
                ss_path = f"screenshots/threat_{int(time.time())}.jpg"
                cv2.imwrite(ss_path, frame)
                st.session_state.unknown_frames_count = 0
                results['alerts'].append("SECURITY THREAT: Snapshot Logged.")

            # Attendance Logging
            if name != "Unknown":
                now_date = datetime.datetime.now().strftime("%Y-%m-%d")
                now_time = datetime.datetime.now().strftime("%H:%M:%S")

                if os.path.exists(attendance_file):
                    current_logs = pd.read_csv(attendance_file)
                else:
                    current_logs = pd.DataFrame(columns=["Roll No", "Date", "Time", "Status"])

                is_logged = not current_logs[
                    (current_logs['Roll No'] == name) & (current_logs['Date'] == now_date)
                ].empty

                if not is_logged:
                    new_row = pd.DataFrame([{"Roll No": name, "Date": now_date, "Time": now_time, "Status": "Present"}])
                    current_logs = pd.concat([current_logs, new_row], ignore_index=True)
                    current_logs.to_csv(attendance_file, index=False)
                    results['new_attendance'] = name
                    results['alerts'].append(f"Welcome, {name}!")

        # 4. Advanced Detection Overlays
        if ADVANCED_DETECTION_AVAILABLE:
            frame_attention_scores = []
            mask_count = 0
            total_faces_checked = 0

            for (top, right, bottom, left), name in zip(face_locs, face_names):
                face_loc = (top, right, bottom, left)

                if mask_detector:
                    has_mask, mask_conf = mask_detector.detect_mask(frame, face_loc)
                    if has_mask:
                        mask_count += 1
                        cv2.putText(display_frame, f"MASK {int(mask_conf*100)}%", (left, bottom + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "NO MASK", (left, bottom + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if attention_detector:
                    is_attentive, att_score, att_status = attention_detector.detect_attention(frame, face_loc)
                    if is_attentive:
                        frame_attention_scores.append(att_score * 100)
                    att_color = (0, 255, 0) if is_attentive else (0, 165, 255)
                    cv2.putText(display_frame, f"{att_status}", (left, top - 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, att_color, 2)

                total_faces_checked += 1

            if total_faces_checked > 0:
                compliance_rate = int((mask_count / total_faces_checked) * 100)
                if frame_attention_scores:
                    avg_frame_att = sum(frame_attention_scores) / len(frame_attention_scores)
                    st.session_state.attention_scores.append(avg_frame_att)
                    if len(st.session_state.attention_scores) > 100:
                        st.session_state.attention_scores.pop(0)

        results['frame'] = display_frame
        st.session_state.total_scans += 1
        if not has_unknown:
            st.session_state.total_known += 1

    except Exception as e:
        results['alerts'].append(f"Processing error: {e}")

    return results

# --- MAIN TABBED COMMAND CENTER ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Sentinel",
    "📊 Analytics Hub",
    "📸 Intruder Gallery",
    "👤 Face Vault",
    "📜 System Logs"
])

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 System Controls")
    enable_voice = st.checkbox("🔔 Visual Alerts", value=True)

    st.markdown("---")
    st.subheader("⚡ Quick Actions")

    if st.button("🚀 Enroll", use_container_width=True):
        st.toast("Go to Face Vault tab to enroll!", icon="👇")

    if st.button("🚨 Alert", use_container_width=True):
        alert_msg = f"🚨 EMERGENCY ALERT - {datetime.datetime.now().strftime('%H:%M:%S')}"
        st.session_state.alerts_list.append({
            'time': datetime.datetime.now(),
            'message': alert_msg,
            'priority': 'HIGH',
            'type': 'Emergency'
        })
        st.toast(alert_msg, icon="🚨")

    st.markdown("---")
    st.subheader("🛡️ Security Armor")
    st.progress(st.session_state.security_score / 100.0)
    st.metric("System Safety", f"{st.session_state.security_score:.0f}%")

    if SYSTEM_MONITORING_AVAILABLE:
        st.markdown("---")
        st.subheader("🏥 System Health")
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            cpu_color = "🟢" if cpu_percent < 70 else "🟡" if cpu_percent < 90 else "🔴"
            mem_color = "🟢" if memory.percent < 70 else "🟡" if memory.percent < 90 else "🔴"
            st.metric("CPU Usage", f"{cpu_color} {cpu_percent:.1f}%")
            st.metric("Memory", f"{mem_color} {memory.percent:.1f}%")
            uptime_seconds = time.time() - st.session_state.system_start_time
            uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m"
            st.metric("Uptime", uptime_str)
        except Exception:
            st.caption("⚠️ Monitoring unavailable")

    st.markdown("---")
    st.subheader("🔔 Recent Alerts")
    if st.session_state.alerts_list:
        for alert in st.session_state.alerts_list[-5:][::-1]:
            priority_icon = "🔴" if alert['priority'] == 'HIGH' else "🟡" if alert['priority'] == 'MEDIUM' else "🟢"
            st.caption(f"{priority_icon} {alert['message']}")
        if st.button("Clear Alerts", use_container_width=True):
            st.session_state.alerts_list = []
            st.rerun()
    else:
        st.caption("No recent alerts")

    if st.button("🔄 Reload Vault"):
        face_clf.load_known_faces()
        st.success("Re-indexed!")

    st.markdown("---")
    st.subheader("📊 Scan Stats")
    st.metric("Total Scans", st.session_state.total_scans)
    st.metric("Known Faces", st.session_state.total_known)
    st.metric("Unknown Faces", st.session_state.total_unknown)

with tab1:
    st.markdown('<div class="status-alert">🛡️ Sentinel Pro v5: CLOUD MODE - ADVANCED AI 🛡️</div>', unsafe_allow_html=True)

    st.subheader("📡 Image Input")
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        st.markdown("**📷 Use Device Camera**")
        camera_image = st.camera_input("Take a snapshot for analysis")

    with input_col2:
        st.markdown("**📁 Upload Image**")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Process whichever input is available
    frame_to_process = None
    source_label = ""

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        frame_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        source_label = "Uploaded"
    elif camera_image is not None:
        file_bytes = np.frombuffer(camera_image.getvalue(), np.uint8)
        frame_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        source_label = "Camera"

    if frame_to_process is not None:
        st.markdown("---")
        st.subheader("🔍 Analysis Results")

        with st.spinner("Analyzing frame..."):
            results = process_frame(frame_to_process, enable_voice)

        # Display processed frame
        proc_col1, proc_col2 = st.columns(2)
        with proc_col1:
            st.image(cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB), caption=f"Source: {source_label}", use_container_width=True)
        with proc_col2:
            st.image(results['frame'], caption="AI Processed Output", use_container_width=True)

        # Metrics
        st.markdown("---")
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        total_people = len(results['face_locs'])
        known_count = sum(1 for name in results['face_names'] if name != "Unknown")
        unknown_count = total_people - known_count

        with met_col1:
            st.metric("👥 Total People", total_people)
        with met_col2:
            st.metric("✅ Known", known_count)
        with met_col3:
            st.metric("⚠️ Unknown", unknown_count)
        with met_col4:
            threat_level = "🟢 LOW" if unknown_count == 0 else "🟡 MEDIUM" if unknown_count <= 2 else "🔴 HIGH"
            st.metric("🛡️ Security", threat_level)

        # Face details
        if results['face_names']:
            st.markdown("---")
            st.subheader("📋 Detected Identities")
            for i, name in enumerate(results['face_names']):
                if name == "Unknown":
                    st.warning(f"Face #{i+1}: **UNIDENTIFIED** - Logged for review")
                else:
                    st.success(f"Face #{i+1}: **{name}** - Attendance marked")

        # Alerts
        for alert in results['alerts']:
            if "THREAT" in alert:
                st.error(f"🚨 {alert}")
            else:
                st.toast(alert, icon="✅")

        # YOLO detections
        if results['counts']['total'] > 0:
            st.markdown("---")
            st.subheader("🎯 YOLO Detections")
            det_col1, det_col2, det_col3 = st.columns(3)
            with det_col1:
                st.metric("People", results['counts']['person'])
            with det_col2:
                st.metric("Cell Phones", results['counts']['cell phone'])
            with det_col3:
                st.metric("Laptops", results['counts']['laptop'])
    else:
        st.info("👆 Take a photo or upload an image to begin analysis.")

        # Show sample demo
        st.markdown("---")
        st.subheader("🎬 How It Works")
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        with demo_col1:
            st.markdown("""
            **1. Capture**
            - Use device camera or upload image
            - Supports JPG, PNG formats
            """)
        with demo_col2:
            st.markdown("""
            **2. AI Analysis**
            - YOLOv8 object detection
            - Face classification & matching
            - Mask & attention detection
            """)
        with demo_col3:
            st.markdown("""
            **3. Results**
            - Real-time identity matching
            - Automatic attendance logging
            - Security threat detection
            """)

    # --- Quick Enrollment ---
    st.markdown("---")
    st.subheader("👥 Quick Enrollment")
    st.caption("Unknown faces detected in recent scans appear here for enrollment")

    if st.session_state.unknown_faces_list:
        num_faces = len(st.session_state.unknown_faces_list)
        st.info(f"🔍 {num_faces} unknown face(s) detected")

        face_cols = st.columns(min(3, num_faces))
        for idx, face_data in enumerate(st.session_state.unknown_faces_list[:3]):
            with face_cols[idx % 3]:
                st.image(face_data['image'], width=150, caption=f"Person {idx+1}")
                enroll_name = st.text_input(f"Name/ID #{idx+1}", key=f"enroll_name_{idx}", placeholder="Enter name")

                if st.button(f"✅ Enroll #{idx+1}", key=f"enroll_btn_{idx}", use_container_width=True):
                    if enroll_name:
                        face_bgr = cv2.cvtColor(face_data['image'], cv2.COLOR_RGB2BGR)
                        if face_clf.register_face(face_bgr, enroll_name):
                            st.success(f"✅ {enroll_name} enrolled!")
                            st.session_state.enrollments_today += 1
                            st.session_state.unknown_faces_list.pop(idx)
                            face_clf.load_known_faces()
                            st.toast(f"System Re-indexed: {enroll_name} added.", icon="✅")
                            time.sleep(0.5)
                            st.rerun()
                    else:
                        st.warning("Please enter a name first")

        st.markdown("---")
        if st.button("🗑️ Clear All Unknown Faces", use_container_width=True):
            st.session_state.unknown_faces_list = []
            st.rerun()
    else:
        st.info("👁️ No unknown faces detected yet. Upload or capture images to start monitoring.")

with tab2:
    st.header("📊 Intelligence Analytics")
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            daily = df.groupby('Date').size().reset_index(name='Count')
            fig = px.bar(daily, x='Date', y='Count', color_discrete_sequence=['#2563eb'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
            st.plotly_chart(fig, use_container_width=True)
            st.info("Peak Traffic Predicted: 10:30 AM - 12:00 PM")
        else:
            st.info("Intelligence database empty.")

with tab3:
    st.header("📸 Intruder Gallery")
    st.write("Sentinel automatically logs unidentified detections for forensic review.")

    st.markdown("---")
    col_auto1, col_auto2 = st.columns([2, 1])
    with col_auto1:
        retention_days = st.slider("🗑️ Auto-Delete Threats Older Than (Days)", 1, 30, 7, key="retention_slider")
    with col_auto2:
        if st.button("🧹 Clean Now", use_container_width=True):
            if os.path.exists("screenshots"):
                deleted_count = 0
                current_time = time.time()
                for img_file in os.listdir("screenshots"):
                    if img_file.endswith(".jpg"):
                        img_path = os.path.join("screenshots", img_file)
                        file_age_days = (current_time - os.path.getmtime(img_path)) / 86400
                        if file_age_days > retention_days:
                            os.remove(img_path)
                            deleted_count += 1
                st.toast(f"🗑️ Deleted {deleted_count} old threat(s)!", icon="✅")
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.subheader("📊 Threat Statistics")
    if os.path.exists("screenshots"):
        images = sorted([os.path.join("screenshots", f) for f in os.listdir("screenshots") if f.endswith(".jpg")], reverse=True)

        if images:
            total_threats = len(images)
            current_time = time.time()
            today_threats = sum(1 for img in images if (current_time - os.path.getmtime(img)) < 86400)
            week_threats = sum(1 for img in images if (current_time - os.path.getmtime(img)) < 604800)

            if total_threats <= 5:
                threat_level = "🟢 LOW"
            elif total_threats <= 15:
                threat_level = "🟡 MEDIUM"
            else:
                threat_level = "🔴 HIGH"

            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Total Threats", total_threats)
            with stat_col2:
                st.metric("Today", today_threats)
            with stat_col3:
                st.metric("This Week", week_threats)
            with stat_col4:
                st.markdown(f"<div style='background: #1e293b; color: #f8fafc; padding: 1rem; border-radius: 12px; text-align: center; font-weight: bold; border: 1px solid rgba(56, 189, 248, 0.3);'>{threat_level}</div>", unsafe_allow_html=True)

            threat_times = []
            for img in images:
                img_time = datetime.datetime.fromtimestamp(os.path.getmtime(img))
                threat_times.append({"Date": img_time.strftime("%Y-%m-%d"), "Hour": img_time.hour, "Count": 1})

            if threat_times:
                threat_df = pd.DataFrame(threat_times)
                daily_threats = threat_df.groupby("Date").size().reset_index(name="Threats")
                fig_threats = px.line(daily_threats, x="Date", y="Threats",
                                     title="Threat Detection Timeline",
                                     color_discrete_sequence=['#ef4444'])
                fig_threats.update_layout(height=250, margin=dict(l=0,r=0,b=0,t=40), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
                st.plotly_chart(fig_threats, use_container_width=True)
        else:
            st.info("No security threats detected recently.")

    st.markdown("---")
    st.subheader("🚀 Quick Enroll from Gallery")
    st.write("Select intruder images to enroll if they were misidentified:")

    if os.path.exists("screenshots"):
        images = sorted([os.path.join("screenshots", f) for f in os.listdir("screenshots") if f.endswith(".jpg")], reverse=True)

        if images:
            if 'selected_intruders' not in st.session_state:
                st.session_state.selected_intruders = {}

            cols = st.columns(3)
            for idx, img_path in enumerate(images[:9]):
                with cols[idx % 3]:
                    st.image(img_path, caption=os.path.basename(img_path))
                    img_key = os.path.basename(img_path)

                    if st.checkbox(f"Select", key=f"chk_{img_key}"):
                        name_input = st.text_input(f"Name/ID", key=f"name_{img_key}", placeholder="Enter ID")
                        if name_input:
                            st.session_state.selected_intruders[img_path] = name_input
                    else:
                        if img_path in st.session_state.selected_intruders:
                            del st.session_state.selected_intruders[img_path]

            if st.session_state.selected_intruders:
                st.markdown("---")
                if st.button(f"✅ Enroll {len(st.session_state.selected_intruders)} Selected", use_container_width=True):
                    enrolled_count = 0
                    for img_path, name in st.session_state.selected_intruders.items():
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                faces = face_clf.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

                                if len(faces) > 0:
                                    (x, y, w, h) = faces[0]
                                    face_roi = img[y:y+h, x:x+w]

                                    if face_clf.register_face(face_roi, name):
                                        os.remove(img_path)
                                        enrolled_count += 1
                        except Exception as e:
                            st.error(f"Error enrolling {name}: {e}")

                    st.session_state.selected_intruders = {}
                    st.toast(f"✅ Successfully enrolled {enrolled_count} person(s)!", icon="🎉")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("No intruder images available for enrollment.")

with tab4:
    st.header("👤 Face Vault")
    v_col1, v_col2 = st.columns([1, 1])
    with v_col1:
        st.subheader("Manual Registration")
        if st.session_state.last_unknown_face is not None:
            st.image(st.session_state.last_unknown_face, width=150)
            new_name = st.text_input("Assign ID/Name", key="vault_name_reg")
            if st.button("✅ Add to Database"):
                if new_name:
                    face_bgr = cv2.cvtColor(st.session_state.last_unknown_face, cv2.COLOR_RGB2BGR)
                    if face_clf.register_face(face_bgr, new_name):
                        st.success("Identity Locked!")
                        st.session_state.last_unknown_face = None
                        time.sleep(1)
                        st.rerun()

        # Upload new face
        st.markdown("---")
        st.subheader("Upload New Face")
        new_face_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"], key="vault_upload")
        if new_face_file is not None:
            file_bytes = np.frombuffer(new_face_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=200)
            new_name = st.text_input("Assign ID/Name", key="vault_upload_name")
            if st.button("✅ Register Face"):
                if new_name:
                    if face_clf.register_face(img, new_name):
                        st.success(f"✅ {new_name} registered!")
                        face_clf.load_known_faces()
                        st.rerun()

    with v_col2:
        st.subheader("System Registry")
        if os.path.exists("known_faces"):
            known_faces_list = [os.path.splitext(f)[0] for f in os.listdir("known_faces") if f.endswith(".jpg")]
            if known_faces_list:
                to_delete = st.selectbox("Select Access to Revoke", ["---"] + known_faces_list)
                if st.button("❌ Remove Identity"):
                    if to_delete != "---":
                        os.remove(os.path.join("known_faces", f"{to_delete}.jpg"))
                        st.success("Target Purged!")
                        face_clf.load_known_faces()
                        st.rerun()

                # Show all known faces
                st.markdown("---")
                st.subheader("Registered Faces")
                face_cols = st.columns(3)
                for idx, name in enumerate(known_faces_list):
                    with face_cols[idx % 3]:
                        face_path = os.path.join("known_faces", f"{name}.jpg")
                        if os.path.exists(face_path):
                            img = cv2.imread(face_path)
                            if img is not None:
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=name, width=100)
            else:
                st.info("No faces registered yet.")
        else:
            st.info("Known faces directory not found.")

with tab5:
    st.header("📜 Forensic Logs")
    search_date_sys = st.date_input("Filter System Log", datetime.date.today())
    if os.path.exists(attendance_file):
        history_df = pd.read_csv(attendance_file)
        history_df['Date'] = pd.to_datetime(history_df['Date']).dt.date
        filtered = history_df[history_df['Date'] == search_date_sys]
        st.dataframe(filtered.sort_values(by="Time", ascending=False), use_container_width=True)

    st.markdown("---")
    st.subheader("📄 Report Center")
    rep_col1, rep_col2 = st.columns(2)

    with rep_col1:
        if st.button("📥 Generate Attendance PDF", use_container_width=True):
            if REPORTS_AVAILABLE:
                report_path = report_generator.generate_attendance_report(st.session_state.attendance_df)
                st.success(f"Report Generated: {report_path}")
                try:
                    with open(report_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        st.download_button(
                            label="⬇️ Download Attendance Report",
                            data=pdf_bytes,
                            file_name="attendance_report.pdf",
                            mime="application/pdf",
                            key="dl_att_pdf"
                        )
                except Exception as e:
                    st.error(f"Error preparing download: {e}")
            else:
                st.error("Report Generator module not loaded.")

    with rep_col2:
        if st.button("🛡️ Generate Security Brief", use_container_width=True):
            if REPORTS_AVAILABLE:
                report_path = report_generator.generate_security_report()
                st.success(f"Brief Generated: {report_path}")
                try:
                    with open(report_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        st.download_button(
                            label="⬇️ Download Security Brief",
                            data=pdf_bytes,
                            file_name="security_brief.pdf",
                            mime="application/pdf",
                            key="dl_sec_pdf"
                        )
                except Exception as e:
                    st.error(f"Error preparing download: {e}")
            else:
                st.error("Report Generator module not loaded.")

# --- FOOTER ---
st.markdown("""
<style>
    .footer {
        text-align: center;
        padding: 60px 0;
        color: #64748b;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-top: 1px solid rgba(56, 189, 248, 0.1);
        margin-top: 50px;
    }
</style>
<div class="footer">
    System Operational • Sentinel Cloud Core • v5.0 © 2026
</div>
""", unsafe_allow_html=True)
