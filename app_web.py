import os
import io
import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from database import Database
from face_engine import FaceEngine
from scanner import PhotoScanner

st.set_page_config(
    page_title="Offline Face Finder",
    page_icon="👤",
    layout="wide"
)

# Initialize paths and singletons
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
candidate_dirs = [
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "_internal", "models"),
    os.path.join(os.getcwd(), "models"),
    "/app/models"
]
MODELS_DIR = next(
    (d for d in candidate_dirs if os.path.exists(os.path.join(d, "face_detection_yunet_2023mar.onnx"))),
    os.path.join(BASE_DIR, "models")
)
DETECTOR_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
DB_DIR = "/data" if os.path.isdir("/data") else BASE_DIR
DB_PATH = os.path.join(DB_DIR, "face_index.db")

@st.cache_resource
def get_engine():
    return FaceEngine(detector_path=DETECTOR_PATH, recognizer_path=RECOGNIZER_PATH)

@st.cache_resource
def get_db():
    return Database(db_path=DB_PATH)

engine = get_engine()
db = get_db()
scanner = PhotoScanner(db=db, engine=engine)

# Header
st.title("👤 Offline Face Finder")
st.caption("🔒 100% Offline • Zero Network Traffic • Works on Any Computer via Docker or Local Python")

# Sidebar: Indexing & Settings
st.sidebar.header("⚙️ 1. Index Photos Folder")

# Default folder path: /photos in Docker, otherwise current directory or Pictures
default_folder = "/photos" if os.path.exists("/photos") else os.path.expanduser("~/Pictures")
folder_to_scan = st.sidebar.text_input("Folder path to scan:", value=default_folder)

stats = db.get_stats()
st.sidebar.metric(label="Indexed Photos", value=f"{stats['photos']:,}")
st.sidebar.metric(label="Indexed Faces", value=f"{stats['faces']:,}")

if st.sidebar.button("⚡ Index / Scan Folder", type="primary", use_container_width=True):
    if not os.path.isdir(folder_to_scan):
        st.sidebar.error(f"Directory not found: `{folder_to_scan}`")
    else:
        progress_bar = st.sidebar.progress(0.0)
        status_text = st.sidebar.empty()

        def progress_cb(current, total, filepath, message):
            if total > 0:
                progress_bar.progress(min(1.0, current / total))
            status_text.text(message)

        scan_stats = scanner.scan_directory(folder_to_scan, progress_callback=progress_cb)
        status_text.text("Indexing completed!")
        st.sidebar.success(f"Done! Processed: {scan_stats['processed']}, Faces added: {scan_stats['faces_found']}")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🎯 2. Search Accuracy")
threshold = st.sidebar.slider(
    "Match Strictness (Cosine Similarity)", 
    min_value=0.30, 
    max_value=0.75, 
    value=0.45, 
    step=0.01,
    help="Higher = Stricter (fewer false positives). Lower = More relaxed (catches different lighting/angles)."
)

pct = int(threshold * 100)
strictness_label = "Relaxed" if pct < 40 else ("Strict" if pct > 55 else "Recommended")
st.sidebar.caption(f"Strictness: **{pct}%** ({strictness_label})")

if st.sidebar.button("🗑 Reset / Clear Database Index", use_container_width=True):
    db.clear_database()
    st.sidebar.warning("Database index cleared!")
    st.rerun()

# Main Search Area
st.subheader("Step 2: Upload Target Person Photo")
uploaded_file = st.file_uploader(
    "Upload a photo of the person you want to find:", 
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    query_img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if query_img_cv is None:
        st.error("Failed to load image.")
    else:
        faces = engine.detect_and_extract_faces(query_img_cv)
        
        if not faces:
            st.warning("⚠️ No clear human face detected in this photo. Please try a clearer picture.")
        else:
            col_preview, col_faces = st.columns([1, 2])
            
            with col_preview:
                st.write("**Uploaded Photo Preview:**")
                st.image(cv2.cvtColor(query_img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col_faces:
                st.write(f"**Found {len(faces)} face(s). Select the target person:**")
                face_options = [f"Person {i+1} (Face detected)" for i in range(len(faces))]
                selected_face_str = st.radio("Choose face:", face_options, horizontal=True)
                selected_idx = face_options.index(selected_face_str)
                selected_face = faces[selected_idx]

                # Show cropped face preview
                bx, by, bw, bh = selected_face["bbox"]
                h, w = query_img_cv.shape[:2]
                margin = int(max(bw, bh) * 0.25)
                crop_x1 = max(0, bx - margin)
                crop_y1 = max(0, by - margin)
                crop_x2 = min(w, bx + bw + margin)
                crop_y2 = min(h, by + bh + margin)
                crop_bgr = query_img_cv[crop_y1:crop_y2, crop_x1:crop_x2]
                st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), caption=f"Selected: Person {selected_idx+1}", width=120)

            st.markdown("---")
            if st.button("🔍 Find All Pictures in System", type="primary", use_container_width=True):
                if stats["photos"] == 0:
                    st.error("⚠️ No photos have been indexed yet. Please index a folder in the sidebar first!")
                else:
                    embeddings_matrix, meta_list = db.load_all_embeddings()
                    
                    t0 = time.time()
                    matches = engine.search_similar(
                        query_embedding=selected_face["embedding"],
                        embeddings_matrix=embeddings_matrix,
                        metadata_list=meta_list,
                        threshold=threshold
                    )
                    latency_ms = (time.time() - t0) * 1000

                    st.subheader(f"Results: Found {len(matches)} matching photo(s) in {latency_ms:.1f}ms")

                    if not matches:
                        st.info("No matching photos found at this threshold level. Try reducing the 'Match Strictness' slider in the sidebar.")
                    else:
                        # Grid view (3 columns)
                        cols = st.columns(3)
                        for idx, match in enumerate(matches):
                            col = cols[idx % 3]
                            with col:
                                filepath = match["filepath"]
                                score_pct = match["similarity_percent"]
                                badge = f"🟢 **{score_pct}% Match**" if score_pct >= 60 else f"🟡 **{score_pct}% Match**"
                                
                                st.markdown(f"### {os.path.basename(filepath)}")
                                st.markdown(badge)
                                st.caption(f"Path: `{filepath}`")
                                
                                if os.path.exists(filepath):
                                    st.image(filepath, use_container_width=True)
                                else:
                                    st.warning("File moved or unreachable.")
