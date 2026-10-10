import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class FaceEngine:
    def __init__(self, 
                 detector_path: str = "models/face_detection_yunet_2023mar.onnx",
                 recognizer_path: str = "models/face_recognition_sface_2021dec.onnx",
                 score_threshold: float = 0.6,
                 nms_threshold: float = 0.3,
                 top_k: int = 5000):
        
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"Detection model not found at {detector_path}")
        if not os.path.exists(recognizer_path):
            raise FileNotFoundError(f"Recognition model not found at {recognizer_path}")
            
        self.detector_path = detector_path
        self.recognizer_path = recognizer_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        
        # Initialize detector with default resolution
        self.detector = cv2.FaceDetectorYN.create(
            self.detector_path,
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k
        )
        self.recognizer = cv2.FaceRecognizerSF.create(self.recognizer_path, "")

    @staticmethod
    def read_image(filepath: str) -> Optional[np.ndarray]:
        """Safely loads an image handling Unicode Windows paths."""
        try:
            if not os.path.exists(filepath):
                return None
            with open(filepath, "rb") as f:
                bytes_arr = bytearray(f.read())
            arr = np.asarray(bytes_arr, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def detect_and_extract_faces(self, img: np.ndarray, max_detect_dim: int = 1280) -> List[Dict]:
        """
        Detects faces in image, aligns each face, and extracts normalized 128-d embeddings.
        Returns list of dicts: [{'bbox': [x, y, w, h], 'score': float, 'embedding': np.ndarray, 'raw_face': face_row}]
        """
        if img is None:
            return []

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return []

        # Downscale for detection if image is massive, preserving aspect ratio
        scale = 1.0
        if max(h, w) > max_detect_dim:
            scale = max_detect_dim / float(max(h, w))
            detect_w = max(int(w * scale), 32)
            detect_h = max(int(h * scale), 32)
            detect_img = cv2.resize(img, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
        else:
            detect_img = img
            detect_w, detect_h = w, h

        self.detector.setInputSize((detect_w, detect_h))
        _, faces = self.detector.detect(detect_img)

        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            # Rescale face coordinates & landmarks back to original image size
            scaled_face = face.copy()
            if scale != 1.0:
                scaled_face[0:14] = scaled_face[0:14] / scale

            # Ensure bounding box is within original bounds
            bx = max(0, int(scaled_face[0]))
            by = max(0, int(scaled_face[1]))
            bw = min(w - bx, int(scaled_face[2]))
            bh = min(h - by, int(scaled_face[3]))
            if bw <= 0 or bh <= 0:
                continue

            try:
                # Align and crop face from original image
                aligned_face = self.recognizer.alignCrop(img, scaled_face)
                if aligned_face is None or aligned_face.size == 0:
                    continue

                # Extract 128-dimensional feature embedding
                feature = self.recognizer.feature(aligned_face).flatten()
                norm = np.linalg.norm(feature)
                if norm > 1e-6:
                    norm_feature = feature / norm
                else:
                    norm_feature = feature

                results.append({
                    "bbox": [bx, by, bw, bh],
                    "score": float(scaled_face[14]),
                    "embedding": norm_feature.astype(np.float32),
                    "raw_face": scaled_face
                })
            except Exception:
                continue

        return results

    def extract_single_face_from_crop_or_image(self, img: np.ndarray) -> Optional[Dict]:
        """
        Extracts face from query image. If multiple faces exist, selects largest/clearest.
        """
        faces = self.detect_and_extract_faces(img)
        if not faces:
            return None
        # Sort by area * score descending
        faces.sort(key=lambda f: (f['bbox'][2] * f['bbox'][3]) * f['score'], reverse=True)
        return faces[0]

    def search_similar(self, 
                       query_embedding: np.ndarray, 
                       embeddings_matrix: np.ndarray, 
                       metadata_list: List[Dict], 
                       threshold: float = 0.45) -> List[Dict]:
        """
        Compares query embedding with all indexed embeddings using vectorized cosine similarity.
        Returns deduplicated matches per photo file, sorted by descending match score.
        """
        if embeddings_matrix.shape[0] == 0 or len(metadata_list) == 0:
            return []

        # Normalize query vector
        q_norm_val = np.linalg.norm(query_embedding)
        q_norm = query_embedding / (q_norm_val + 1e-7) if q_norm_val > 1e-7 else query_embedding

        # Normalize database vectors
        mat_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        mat_norms[mat_norms == 0] = 1.0
        normalized_matrix = embeddings_matrix / mat_norms

        # Vectorized dot-product: (N, 128) x (128,) -> (N,)
        cosine_scores = np.dot(normalized_matrix, q_norm)

        # Group matches by filepath, keeping the highest score for each photo
        photo_best_matches = {}
        for idx, score in enumerate(cosine_scores):
            bounded_score = float(np.clip(score, -1.0, 1.0))
            if bounded_score >= threshold:
                meta = metadata_list[idx]
                filepath = meta["filepath"]
                if filepath not in photo_best_matches or bounded_score > photo_best_matches[filepath]["score"]:
                    photo_best_matches[filepath] = {
                        "filepath": filepath,
                        "score": bounded_score,
                        "similarity_percent": round(max(0.0, bounded_score) * 100, 1),
                        "bbox": meta["bbox"]
                    }

        sorted_matches = sorted(photo_best_matches.values(), key=lambda m: m["score"], reverse=True)
        return sorted_matches
