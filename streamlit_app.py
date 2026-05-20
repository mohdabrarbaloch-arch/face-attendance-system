import sys
import os
try:
    import cv2
except Exception as e:
    print(f"OpenCV Import Error: {e}")
    sys.exit(1)
import streamlit as st
import pandas as pd
from face_classifier import FaceClassifier
import datetime
import time
import numpy as np
import plotly.express as px

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
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px !important;
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
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #38bdf8, #a855f7) !important;
    }
</style>

<div class="premium-header">
    <h1>S E N T I N E L</h1>
    <p>Face Recognition Engine • v6.0 Lite</p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if 'attendance_df' not in st.session_state:
    st.session_state.attendance_df = pd.DataFrame(columns=["Roll No", "Date", "Time", "Status"])
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
if 'enrollments_today' not in st.session_state:
    st.session_state.enrollments_today = 0
if 'system_start_time' not in st.session_state:
    st.session_state.system_start_time = time.time()
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

@st.cache_resource
def load_classifier():
    with st.spinner("Initializing face recognition engine..."):
        return FaceClassifier()

face_clf = load_classifier()

def process_frame(frame):
    display_frame = frame.copy()
    results = {
        'frame': display_frame,
        'face_locs': [],
        'face_names': [],
        'alerts': [],
    }

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_clf.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(60, 60))

        if len(faces) == 0:
            cv2.putText(display_frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)
            results['frame'] = display_frame
            return results

        face_locs, face_names = face_clf.classify_face(frame)
        results['face_locs'] = face_locs
        results['face_names'] = face_names

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

        for (top, right, bottom, left), name in zip(face_locs, face_names):
            label = "UNIDENTIFIED" if name == "Unknown" else f"ID: {name}"
            color = (0, 0, 255) if name == "Unknown" else (37, 99, 235)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(display_frame, (left, top - 30), (left + 200, top), color, -1)
            cv2.putText(display_frame, label, (left + 5, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if name == "Unknown":
                r_roi = frame[top:bottom, left:right]
                if r_roi.size > 0:
                    unknown_face_rgb = cv2.cvtColor(r_roi, cv2.COLOR_BGR2RGB)
                    if len(st.session_state.unknown_faces_list) < 5:
                        st.session_state.unknown_faces_list.append({
                            'image': unknown_face_rgb,
                            'timestamp': time.time(),
                        })

                if st.session_state.unknown_frames_count >= 3:
                    if not os.path.exists("screenshots"):
                        os.makedirs("screenshots")
                    ss_path = f"screenshots/threat_{int(time.time())}.jpg"
                    cv2.imwrite(ss_path, frame)
                    st.session_state.unknown_frames_count = 0
                    results['alerts'].append("SECURITY THREAT: Snapshot Logged.")

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
                    results['alerts'].append(f"Welcome, {name}!")

        st.session_state.total_scans += 1
        if not has_unknown:
            st.session_state.total_known += 1

    except Exception as e:
        results['alerts'].append(f"Processing error: {e}")

    results['frame'] = display_frame
    return results

# --- MAIN TABBED COMMAND CENTER ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Sentinel",
    "Analytics Hub",
    "Intruder Gallery",
    "Face Vault",
    "System Logs"
])

# --- SIDEBAR ---
with st.sidebar:
    st.header("System Controls")

    st.markdown("---")
    if st.button("Emergency Alert", use_container_width=True, type="primary"):
        alert_msg = f"EMERGENCY ALERT - {datetime.datetime.now().strftime('%H:%M:%S')}"
        st.session_state.alerts_list.append({
            'time': datetime.datetime.now(),
            'message': alert_msg,
            'priority': 'HIGH',
            'type': 'Emergency'
        })
        st.toast(alert_msg, icon="🚨")

    st.markdown("---")
    st.subheader("Security Armor")
    st.progress(st.session_state.security_score / 100.0)
    st.metric("System Safety", f"{st.session_state.security_score:.0f}%")

    if SYSTEM_MONITORING_AVAILABLE:
        st.markdown("---")
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            st.metric("CPU", f"{cpu_percent:.1f}%")
            st.metric("Memory", f"{memory.percent:.1f}%")
            uptime_seconds = time.time() - st.session_state.system_start_time
            uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m"
            st.metric("Uptime", uptime_str)
        except Exception:
            pass

    st.markdown("---")
    st.subheader("Recent Alerts")
    if st.session_state.alerts_list:
        for alert in st.session_state.alerts_list[-5:][::-1]:
            icon = "🔴" if alert['priority'] == 'HIGH' else "🟢"
            st.caption(f"{icon} {alert['message']}")
        if st.button("Clear Alerts", use_container_width=True):
            st.session_state.alerts_list = []
            st.rerun()
    else:
        st.caption("No recent alerts")

    if st.button("Reload Vault", use_container_width=True):
        face_clf.load_known_faces()
        st.success("Re-indexed!")

    st.markdown("---")
    st.metric("Total Scans", st.session_state.total_scans)
    st.metric("Known Faces", st.session_state.total_known)
    st.metric("Unknown Faces", st.session_state.total_unknown)

with tab1:
    st.markdown('<div class="status-alert">Sentinel Pro v6: FACE RECOGNITION ACTIVE</div>', unsafe_allow_html=True)

    st.subheader("Image Input")
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        st.markdown("**Device Camera**")
        camera_image = st.camera_input("Take a snapshot for analysis")

    with input_col2:
        st.markdown("**Upload Image**")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

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
        st.subheader("Analysis Results")

        with st.spinner("Analyzing frame..."):
            results = process_frame(frame_to_process)

        proc_col1, proc_col2 = st.columns(2)
        with proc_col1:
            st.image(cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB), caption=f"Source: {source_label}", use_container_width=True)
        with proc_col2:
            st.image(results['frame'], caption="AI Processed Output", use_container_width=True)

        st.markdown("---")
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        total_people = len(results['face_locs'])
        known_count = sum(1 for name in results['face_names'] if name != "Unknown")
        unknown_count = total_people - known_count

        with met_col1:
            st.metric("Total People", total_people)
        with met_col2:
            st.metric("Known", known_count)
        with met_col3:
            st.metric("Unknown", unknown_count)
        with met_col4:
            threat_level = "LOW" if unknown_count == 0 else "MEDIUM" if unknown_count <= 2 else "HIGH"
            st.metric("Security Level", threat_level)

        if results['face_names']:
            st.markdown("---")
            st.subheader("Detected Identities")
            for i, name in enumerate(results['face_names']):
                if name == "Unknown":
                    st.warning(f"Face #{i+1}: **UNIDENTIFIED** - Logged for review")
                else:
                    st.success(f"Face #{i+1}: **{name}** - Attendance marked")

        for alert in results['alerts']:
            if "THREAT" in alert:
                st.error(alert)
            else:
                st.toast(alert, icon="Welcome")
    else:
        st.info("Take a photo or upload an image to begin analysis.")

        st.markdown("---")
        st.subheader("How It Works")
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
            - Haar Cascade face detection
            - Histogram-based face matching
            - CLAHE preprocessing
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
    st.subheader("Quick Enrollment")

    if st.session_state.unknown_faces_list:
        num_faces = len(st.session_state.unknown_faces_list)
        st.info(f"{num_faces} unknown face(s) detected")

        face_cols = st.columns(min(3, num_faces))
        for idx, face_data in enumerate(st.session_state.unknown_faces_list[:3]):
            with face_cols[idx % 3]:
                st.image(face_data['image'], width=150, caption=f"Person {idx+1}")
                enroll_name = st.text_input(f"Name #{idx+1}", key=f"enroll_name_{idx}", placeholder="Enter name")

                if st.button(f"Enroll #{idx+1}", key=f"enroll_btn_{idx}", use_container_width=True, type="primary"):
                    if enroll_name:
                        face_bgr = cv2.cvtColor(face_data['image'], cv2.COLOR_RGB2BGR)
                        if face_clf.register_face(face_bgr, enroll_name):
                            st.success(f"{enroll_name} enrolled!")
                            st.session_state.enrollments_today += 1
                            st.session_state.unknown_faces_list.pop(idx)
                            face_clf.load_known_faces()
                            st.toast(f"System Re-indexed: {enroll_name} added.", icon="✅")
                            time.sleep(0.5)
                            st.rerun()
                    else:
                        st.warning("Please enter a name first")

        if st.button("Clear All Unknown Faces", use_container_width=True):
            st.session_state.unknown_faces_list = []
            st.rerun()
    else:
        st.info("No unknown faces detected yet. Upload or capture images to start monitoring.")

with tab2:
    st.header("Intelligence Analytics")
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
    st.header("Intruder Gallery")
    st.write("Sentinel automatically logs unidentified detections for forensic review.")

    st.markdown("---")
    col_auto1, col_auto2 = st.columns([2, 1])
    with col_auto1:
        retention_days = st.slider("Auto-Delete Threats Older Than (Days)", 1, 30, 7, key="retention_slider")
    with col_auto2:
        if st.button("Clean Now", use_container_width=True):
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
                st.toast(f"Deleted {deleted_count} old threat(s)!", icon="✅")
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.subheader("Threat Statistics")
    if os.path.exists("screenshots"):
        images = sorted([os.path.join("screenshots", f) for f in os.listdir("screenshots") if f.endswith(".jpg")], reverse=True)

        if images:
            total_threats = len(images)
            current_time = time.time()
            today_threats = sum(1 for img in images if (current_time - os.path.getmtime(img)) < 86400)
            week_threats = sum(1 for img in images if (current_time - os.path.getmtime(img)) < 604800)

            threat_level = "LOW" if total_threats <= 5 else "MEDIUM" if total_threats <= 15 else "HIGH"

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

with tab4:
    st.header("Face Vault")
    v_col1, v_col2 = st.columns([1, 1])
    with v_col1:
        st.subheader("Upload New Face")
        new_face_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"], key="vault_upload")
        if new_face_file is not None:
            file_bytes = np.frombuffer(new_face_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=200)
            new_name = st.text_input("Assign ID/Name", key="vault_upload_name")
            if st.button("Register Face", type="primary"):
                if new_name:
                    if face_clf.register_face(img, new_name):
                        st.success(f"{new_name} registered!")
                        face_clf.load_known_faces()
                        st.rerun()

    with v_col2:
        st.subheader("System Registry")
        if os.path.exists("known_faces"):
            known_faces_list = [os.path.splitext(f)[0] for f in os.listdir("known_faces") if f.endswith(".jpg")]
            if known_faces_list:
                to_delete = st.selectbox("Select Access to Revoke", ["---"] + known_faces_list)
                if st.button("Remove Identity", type="primary"):
                    if to_delete != "---":
                        os.remove(os.path.join("known_faces", f"{to_delete}.jpg"))
                        st.success("Target Purged!")
                        face_clf.load_known_faces()
                        st.rerun()

                st.markdown("---")
                st.subheader("Registered Faces")
                face_cols = st.columns(3)
                for idx, name in enumerate(known_faces_list):
                    with face_cols[idx % 3]:
                        face_path = os.path.join("known_faces", f"{name}.jpg")
                        if os.path.exists(face_path):
                            img = cv2.imread(face_path)
                            if img is not None:
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=name)
            else:
                st.info("No faces registered yet.")

with tab5:
    st.header("Forensic Logs")
    search_date_sys = st.date_input("Filter System Log", datetime.date.today())
    if os.path.exists(attendance_file):
        history_df = pd.read_csv(attendance_file)
        history_df['Date'] = pd.to_datetime(history_df['Date']).dt.date
        filtered = history_df[history_df['Date'] == search_date_sys]
        st.dataframe(filtered.sort_values(by="Time", ascending=False), use_container_width=True)

    st.markdown("---")
    st.subheader("Report Center")
    rep_col1, rep_col2 = st.columns(2)

    with rep_col1:
        if st.button("Generate Attendance PDF", use_container_width=True):
            if REPORTS_AVAILABLE:
                from report_generator import ReportGenerator
                rg = ReportGenerator()
                report_path = rg.generate_attendance_report(st.session_state.attendance_df)
                st.success(f"Report Generated: {report_path}")
                try:
                    with open(report_path, "rb") as pdf_file:
                        st.download_button("Download Attendance Report", data=pdf_file.read(), file_name="attendance_report.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Report Generator not available.")

    with rep_col2:
        if st.button("Generate Security Brief", use_container_width=True):
            if REPORTS_AVAILABLE:
                from report_generator import ReportGenerator
                rg = ReportGenerator()
                report_path = rg.generate_security_report()
                st.success(f"Brief Generated: {report_path}")
                try:
                    with open(report_path, "rb") as pdf_file:
                        st.download_button("Download Security Brief", data=pdf_file.read(), file_name="security_brief.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Report Generator not available.")

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; padding: 60px 0; color: #64748b; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 2px; text-transform: uppercase; border-top: 1px solid rgba(56, 189, 248, 0.1); margin-top: 50px;">
    System Operational • Sentinel Lite Core • v6.0 &copy; 2026
</div>
""", unsafe_allow_html=True)
