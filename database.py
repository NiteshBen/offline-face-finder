import os
import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Optional

class Database:
    def __init__(self, db_path: str = "face_index.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT UNIQUE NOT NULL,
                    file_size INTEGER NOT NULL,
                    modified_time REAL NOT NULL,
                    face_count INTEGER DEFAULT 0,
                    indexed_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    photo_id INTEGER NOT NULL,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_w INTEGER NOT NULL,
                    bbox_h INTEGER NOT NULL,
                    score REAL NOT NULL,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_filepath ON photos(filepath)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id)")
            conn.commit()

    def get_indexed_photo_info(self, filepath: str) -> Optional[Tuple[int, float]]:
        """Returns (file_size, modified_time) if already indexed, else None."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT file_size, modified_time FROM photos WHERE filepath = ?", (filepath,))
            row = cur.fetchone()
            if row:
                return (row[0], row[1])
            return None

    def save_photo_faces(self, filepath: str, file_size: int, modified_time: float, 
                         indexed_at: float, faces_data: List[Dict]):
        """
        faces_data: list of dicts with keys:
          'bbox': [x, y, w, h], 'score': float, 'embedding': np.ndarray (128,)
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            # If photo existed, remove old faces via cascade or delete
            cur.execute("DELETE FROM photos WHERE filepath = ?", (filepath,))
            
            cur.execute("""
                INSERT INTO photos (filepath, file_size, modified_time, face_count, indexed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (filepath, file_size, modified_time, len(faces_data), indexed_at))
            photo_id = cur.lastrowid

            for f in faces_data:
                bbox = f['bbox']
                emb_bytes = f['embedding'].astype(np.float32).tobytes()
                cur.execute("""
                    INSERT INTO faces (photo_id, bbox_x, bbox_y, bbox_w, bbox_h, score, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (photo_id, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), 
                      float(f.get('score', 1.0)), emb_bytes))
            conn.commit()

    def remove_missing_files(self, existing_filepaths: set) -> int:
        """Removes records for files that no longer exist on disk."""
        removed_count = 0
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, filepath FROM photos")
            rows = cur.fetchall()
            for photo_id, filepath in rows:
                if filepath not in existing_filepaths or not os.path.exists(filepath):
                    cur.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
                    removed_count += 1
            conn.commit()
        return removed_count

    def get_stats(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM photos")
            photo_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM faces")
            face_count = cur.fetchone()[0]
            return {"photos": photo_count, "faces": face_count}

    def load_all_embeddings(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Loads all face embeddings into a 2D numpy array (N, 128)
        and returns a matching list of metadata dictionaries.
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, f.embedding, p.filepath
                FROM faces f
                JOIN photos p ON f.photo_id = p.id
            """)
            rows = cur.fetchall()

        if not rows:
            return np.empty((0, 128), dtype=np.float32), []

        embeddings_list = []
        meta_list = []

        for row in rows:
            face_id, bx, by, bw, bh, emb_blob, filepath = row
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            # Ensure 128 elements
            if emb.shape[0] == 128:
                embeddings_list.append(emb)
                meta_list.append({
                    "face_id": face_id,
                    "filepath": filepath,
                    "bbox": (bx, by, bw, bh)
                })

        if not embeddings_list:
            return np.empty((0, 128), dtype=np.float32), []

        embeddings_matrix = np.vstack(embeddings_list)
        return embeddings_matrix, meta_list

    def clear_database(self):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM faces")
            conn.execute("DELETE FROM photos")
            conn.commit()
