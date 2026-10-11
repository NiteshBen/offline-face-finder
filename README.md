# Offline Face Finder 👤

A lightweight, 100% offline desktop application that scans your local computer's photos and finds all pictures matching a specific person.

---

### Features
* **100% Offline & Private**: Zero internet connection required. No images or data ever leave your PC.
* **Fast Incremental Indexing**: Scans your local photos folder once and saves face representations into a local SQLite database (`face_index.db`). Future scans skip unchanged files automatically.
* **Instant Search (<0.05 seconds)**: Uses vectorized Cosine Similarity over 128-dimensional deep face embeddings to find matches across thousands of photos in milliseconds.
* **Multi-Face Handling**: If the uploaded query picture contains multiple people, the UI lets you select exactly which person you want to search for.
* **Native Windows Integration**:
  * **Open Photo**: Opens the matched picture in your default Windows photo viewer.
  * **Show in Folder**: Opens Windows File Explorer with the exact matched photo highlighted in blue.

---

### How to Run

#### Option 1: Run on Any System via Docker 🐳 (Windows, Mac, Linux)
The ultimate cross-platform method that guarantees it runs on any computer without missing dependencies:
```bash
# 1. Build the Docker image
docker build -t offline-face-finder .

# 2. Run container (mount your photos folder to /photos)
# Windows PowerShell:
docker run -p 8501:8501 -v "C:\Users\YourName\Pictures:/photos" -v "${PWD}/data:/data" offline-face-finder

# Linux / Mac:
docker run -p 8501:8501 -v "$HOME/Pictures:/photos" -v "$(pwd)/data:/data" offline-face-finder
```
Then open **`http://localhost:8501`** in any web browser!

Or simply use **Docker Compose**:
```bash
docker compose up
```

#### Option 2: Portable Distribution for Windows (`OfflineFaceFinder_portable.zip`)
To send this project to another Windows computer without installing Python or Docker:
1. Extract **`OfflineFaceFinder_portable.zip`**.
2. Open the extracted folder and double-click **`OfflineFaceFinder_portable.exe`**.
*(Note: If a target Windows PC is completely fresh and lacks the standard C++ runtime, install the official [Microsoft Visual C++ 2015–2022 x64 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)).*

#### Option 3: Local Python Launcher
```powershell
.\.venv\Scripts\python.exe main.py
# Or for the browser interface:
.\.venv\Scripts\streamlit.exe run app_web.py
```

---

### How to Use

1. **Step 1: Scan / Index Your Photos**:
   * Click **"Select Photos Folder"** (for example: `C:\Users\YourName\Pictures` or `D:\FamilyPhotos`).
   * Click **"Index / Scan Folder"**. A progress bar will track each photo being processed and faces stored.
2. **Step 2: Search for a Person**:
   * Click **"Choose Photo of Person"** and select a photo of the person you want to find.
   * If there are multiple people in the picture, click **Face 1**, **Face 2**, etc., to choose which person to search.
   * Adjust the **"Search Accuracy"** slider if needed (45% is recommended; higher = stricter, lower = more relaxed).
   * Click **"Find All Pictures in System"**.
3. **Step 3: View Results**:
   * The app displays a gallery of all matching pictures with their confidence scores.
   * Click **"Open Photo"** to view it or **"Show in Folder"** to jump straight to its location in Windows File Explorer.

---

### File Structure
* `models/` - Contains local pre-trained AI models (`face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`).
* `database.py` - Local SQLite storage engine for fast vector lookups.
* `face_engine.py` - OpenCV DNN facial detection, alignment, embedding extraction, and cosine similarity search.
* `scanner.py` - Background recursive folder scanner with incremental caching.
* `app_gui.py` - Modern native desktop GUI built with CustomTkinter.
* `main.py` - Application entry point.
* `run.bat` - Double-clickable batch launcher.
