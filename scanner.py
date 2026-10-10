import os
import time
import threading
from typing import Callable, Optional
from database import Database
from face_engine import FaceEngine

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

class PhotoScanner:
    def __init__(self, db: Database, engine: FaceEngine):
        self.db = db
        self.engine = engine
        self._stop_requested = threading.Event()
        self.is_running = False

    def stop(self):
        self._stop_requested.set()

    def scan_directory(self, 
                       root_dir: str, 
                       progress_callback: Optional[Callable[[int, int, str, str], None]] = None) -> dict:
        """
        Recursively scans directory for photos and indexes faces.
        progress_callback signature: (current_idx, total_files, current_filepath, status_message)
        """
        self._stop_requested.clear()
        self.is_running = True

        stats = {
            "total_found": 0,
            "processed": 0,
            "skipped": 0,
            "faces_found": 0,
            "errors": 0,
            "stopped": False
        }

        try:
            # 1. Discover all image files
            image_paths = []
            for dirpath, _, filenames in os.walk(root_dir):
                if self._stop_requested.is_set():
                    break
                for fname in filenames:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        full_path = os.path.normpath(os.path.join(dirpath, fname))
                        image_paths.append(full_path)

            total_files = len(image_paths)
            stats["total_found"] = total_files

            if total_files == 0:
                if progress_callback:
                    progress_callback(0, 0, "", "No supported image files found in folder.")
                return stats

            # 2. Process each image
            for idx, img_path in enumerate(image_paths):
                if self._stop_requested.is_set():
                    stats["stopped"] = True
                    break

                try:
                    fsize = os.path.getsize(img_path)
                    mtime = os.path.getmtime(img_path)
                except Exception:
                    stats["errors"] += 1
                    continue

                # Check if already indexed and unchanged
                existing = self.db.get_indexed_photo_info(img_path)
                if existing is not None:
                    cached_size, cached_mtime = existing
                    if cached_size == fsize and abs(cached_mtime - mtime) < 1.0:
                        stats["skipped"] += 1
                        if progress_callback and idx % 10 == 0:
                            progress_callback(idx + 1, total_files, img_path, f"Skipping unchanged: {os.path.basename(img_path)}")
                        continue

                # Process photo
                if progress_callback:
                    progress_callback(idx + 1, total_files, img_path, f"Scanning: {os.path.basename(img_path)}")

                try:
                    img = self.engine.read_image(img_path)
                    if img is None:
                        stats["errors"] += 1
                        continue

                    faces = self.engine.detect_and_extract_faces(img)
                    self.db.save_photo_faces(
                        filepath=img_path,
                        file_size=fsize,
                        modified_time=mtime,
                        indexed_at=time.time(),
                        faces_data=faces
                    )

                    stats["processed"] += 1
                    stats["faces_found"] += len(faces)

                except Exception:
                    stats["errors"] += 1

            if progress_callback:
                if stats["stopped"]:
                    progress_callback(stats["processed"], total_files, "", "Scanning stopped by user.")
                else:
                    progress_callback(total_files, total_files, "", "Indexing complete!")

        finally:
            self.is_running = False

        return stats
