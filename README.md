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

#### Option 1: Standalone Executable (Recommended)
Simply double-click:
```
OfflineFaceFinder.exe
```
You don't need Python installed or any terminal commands. You can even create a shortcut to this `.exe` directly on your Windows Desktop!

#### Option 2: One-Click Batch Launcher
Double-click `run.bat` in this folder.

#### Option 3: Via Python Terminal
```powershell
.\.venv\Scripts\python.exe main.py
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
