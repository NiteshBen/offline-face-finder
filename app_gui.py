import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from database import Database
from face_engine import FaceEngine
from scanner import PhotoScanner

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class OfflineFaceFinderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Offline Face Finder — 100% Local & Private")
        self.geometry("1100x750")
        self.minsize(950, 650)

        # Initialize core components (handles both Python script and frozen .exe)
        if getattr(sys, 'frozen', False):
            bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            app_dir = os.path.dirname(sys.executable)
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
            app_dir = bundle_dir

        models_dir = os.path.join(bundle_dir, "models")
        detector_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        recognizer_path = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")
        db_path = os.path.join(app_dir, "face_index.db")

        try:
            self.engine = FaceEngine(detector_path=detector_path, recognizer_path=recognizer_path)
            self.db = Database(db_path=db_path)
            self.scanner = PhotoScanner(db=self.db, engine=self.engine)
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to load AI models:\n{e}")
            sys.exit(1)

        # State
        self.query_image_path: Optional[str] = None
        self.query_image_cv: Optional[np.ndarray] = None
        self.query_faces: List[Dict] = []
        self.selected_face_idx: int = 0
        self.search_results: List[Dict] = []
        self.thumbnail_cache = {}  # Keep references so Tkinter doesn't garbage collect

        self._build_ui()
        self._update_stats_display()

    def _build_ui(self):
        # Configure grid layout: 2 columns (Sidebar: 340px, Main: expand)
        self.grid_columnconfigure(0, weight=0, minsize=340)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # LEFT SIDEBAR: Indexing & Settings
        # ==========================================
        self.sidebar = ctk.CTkFrame(self, width=340, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_propagate(False)

        # App Brand Header
        title_label = ctk.CTkLabel(
            self.sidebar, 
            text="👤 Face Finder", 
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_label.pack(padx=20, pady=(20, 4), anchor="w")

        privacy_badge = ctk.CTkLabel(
            self.sidebar,
            text="🔒 100% Offline • Zero Internet Used",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#4ade80"
        )
        privacy_badge.pack(padx=20, pady=(0, 15), anchor="w")

        # Stats Card
        self.stats_frame = ctk.CTkFrame(self.sidebar, fg_color="#1e293b", corner_radius=8)
        self.stats_frame.pack(fill="x", padx=15, pady=5)
        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="Indexed Photos: 0\nIndexed Faces: 0",
            font=ctk.CTkFont(size=13),
            justify="left"
        )
        self.stats_label.pack(padx=12, pady=10, anchor="w")

        # Indexing Section
        idx_heading = ctk.CTkLabel(
            self.sidebar, 
            text="Step 1: Scan Your Photos", 
            font=ctk.CTkFont(size=15, weight="bold")
        )
        idx_heading.pack(padx=15, pady=(15, 6), anchor="w")

        self.btn_select_folder = ctk.CTkButton(
            self.sidebar, 
            text="📁 Select Photos Folder", 
            command=self._on_select_folder,
            fg_color="#334155",
            hover_color="#475569"
        )
        self.btn_select_folder.pack(fill="x", padx=15, pady=4)

        self.folder_label = ctk.CTkLabel(
            self.sidebar, 
            text="No folder selected", 
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=300,
            justify="left"
        )
        self.folder_label.pack(padx=15, pady=(0, 6), anchor="w")

        self.btn_scan = ctk.CTkButton(
            self.sidebar, 
            text="⚡ Index / Scan Folder", 
            command=self._on_start_scan,
            state="disabled"
        )
        self.btn_scan.pack(fill="x", padx=15, pady=4)

        self.btn_stop = ctk.CTkButton(
            self.sidebar, 
            text="⏹ Stop Scanning", 
            command=self._on_stop_scan,
            fg_color="#b91c1c",
            hover_color="#dc2626",
            state="disabled"
        )
        self.btn_stop.pack(fill="x", padx=15, pady=4)

        self.progress_bar = ctk.CTkProgressBar(self.sidebar)
        self.progress_bar.pack(fill="x", padx=15, pady=(8, 4))
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(
            self.sidebar, 
            text="Ready", 
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=300,
            justify="left"
        )
        self.status_label.pack(padx=15, pady=(0, 10), anchor="w")

        # Threshold Settings
        settings_heading = ctk.CTkLabel(
            self.sidebar, 
            text="Search Accuracy", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        settings_heading.pack(padx=15, pady=(10, 4), anchor="w")

        self.threshold_slider = ctk.CTkSlider(
            self.sidebar, 
            from_=0.30, 
            to=0.75, 
            number_of_steps=45,
            command=self._on_threshold_change
        )
        self.threshold_slider.pack(fill="x", padx=15, pady=4)
        self.threshold_slider.set(0.45)

        self.threshold_label = ctk.CTkLabel(
            self.sidebar, 
            text="Strictness: 45% (Recommended)", 
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.threshold_label.pack(padx=15, pady=(0, 10), anchor="w")

        # Clear Database Button at bottom
        self.btn_clear_db = ctk.CTkButton(
            self.sidebar, 
            text="🗑 Clear Photo Index", 
            command=self._on_clear_index,
            fg_color="transparent",
            border_width=1,
            border_color="#475569",
            text_color="#94a3b8",
            hover_color="#1e293b"
        )
        self.btn_clear_db.pack(side="bottom", fill="x", padx=15, pady=15)

        # ==========================================
        # RIGHT MAIN PANEL: Query & Results
        # ==========================================
        self.main_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.main_panel.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.main_panel.grid_rowconfigure(1, weight=1)
        self.main_panel.grid_columnconfigure(0, weight=1)

        # Top Query Section Card
        self.query_card = ctk.CTkFrame(self.main_panel, corner_radius=10)
        self.query_card.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))

        query_title = ctk.CTkLabel(
            self.query_card, 
            text="Step 2: Upload Target Person Photo", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        query_title.pack(padx=15, pady=(10, 4), anchor="w")

        # Query content row
        self.query_row = ctk.CTkFrame(self.query_card, fg_color="transparent")
        self.query_row.pack(fill="x", padx=15, pady=(0, 10))

        # Query Preview Box
        self.query_preview_label = ctk.CTkLabel(
            self.query_row, 
            text="No Photo Selected\nClick 'Choose Photo' below",
            width=140, 
            height=140, 
            fg_color="#1e293b", 
            corner_radius=8
        )
        self.query_preview_label.pack(side="left", padx=(0, 15), pady=5)

        # Controls next to preview
        self.query_controls = ctk.CTkFrame(self.query_row, fg_color="transparent")
        self.query_controls.pack(side="left", fill="both", expand=True)

        self.btn_upload = ctk.CTkButton(
            self.query_controls, 
            text="📷 Choose Photo of Person", 
            command=self._on_upload_query,
            height=36,
            font=ctk.CTkFont(weight="bold")
        )
        self.btn_upload.pack(anchor="w", pady=(5, 6))

        self.face_select_info = ctk.CTkLabel(
            self.query_controls, 
            text="Upload an image containing the face to find.", 
            font=ctk.CTkFont(size=12),
            text_color="gray",
            justify="left"
        )
        self.face_select_info.pack(anchor="w", pady=(0, 6))

        self.face_selector_row = ctk.CTkFrame(self.query_controls, fg_color="transparent")
        self.face_selector_row.pack(anchor="w", pady=(0, 6))

        self.btn_search = ctk.CTkButton(
            self.query_controls, 
            text="🔍 Find All Pictures in System", 
            command=self._on_search,
            fg_color="#16a34a",
            hover_color="#15803d",
            height=36,
            font=ctk.CTkFont(weight="bold"),
            state="disabled"
        )
        self.btn_search.pack(anchor="w", pady=(5, 0))

        # Bottom Results Section
        self.results_header_frame = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        self.results_header_frame.grid(row=1, column=0, sticky="ew", pady=(5, 5))
        
        self.results_title = ctk.CTkLabel(
            self.results_header_frame, 
            text="Matches Found: 0", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.results_title.pack(side="left")

        # Scrollable Frame for Results
        self.results_scroll = ctk.CTkScrollableFrame(
            self.main_panel, 
            corner_radius=8, 
            fg_color="#0f172a"
        )
        self.results_scroll.grid(row=2, column=0, sticky="nsew", pady=(0, 5))
        self.main_panel.grid_rowconfigure(2, weight=1)

        self._show_empty_results_placeholder("Upload a photo and click 'Find All Pictures' to view results.")

    # ==========================================
    # LOGIC & EVENT HANDLERS
    # ==========================================
    def _update_stats_display(self):
        stats = self.db.get_stats()
        self.stats_label.configure(
            text=f"Indexed Photos: {stats['photos']:,}\nIndexed Faces:   {stats['faces']:,}"
        )

    def _on_threshold_change(self, value):
        pct = int(value * 100)
        desc = "Relaxed" if pct < 40 else ("Strict" if pct > 55 else "Recommended")
        self.threshold_label.configure(text=f"Strictness: {pct}% ({desc})")

    def _on_select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder With Photos")
        if folder:
            self.selected_folder = folder
            self.folder_label.configure(text=folder)
            self.btn_scan.configure(state="normal")

    def _on_start_scan(self):
        if not hasattr(self, 'selected_folder') or not self.selected_folder:
            return

        self.btn_scan.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_select_folder.configure(state="disabled")
        self.progress_bar.set(0)

        # Run scanner in background thread to keep GUI responsive
        threading.Thread(target=self._run_scan_thread, daemon=True).start()

    def _run_scan_thread(self):
        def progress_cb(current, total, current_file, message):
            self.after(0, self._update_scan_progress, current, total, current_file, message)

        self.scanner.scan_directory(self.selected_folder, progress_callback=progress_cb)
        self.after(0, self._scan_finished)

    def _update_scan_progress(self, current, total, current_file, message):
        if total > 0:
            frac = current / total
            self.progress_bar.set(frac)
        self.status_label.configure(text=message)
        self._update_stats_display()

    def _scan_finished(self):
        self.btn_scan.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_select_folder.configure(state="normal")
        self._update_stats_display()
        messagebox.showinfo("Scan Complete", "Photo folder indexing has finished!")

    def _on_stop_scan(self):
        self.scanner.stop()
        self.btn_stop.configure(state="disabled")
        self.status_label.configure(text="Stopping scanner...")

    def _on_clear_index(self):
        confirm = messagebox.askyesno(
            "Clear Database", 
            "Are you sure you want to delete the local photo index?\nThis does NOT delete your photo files."
        )
        if confirm:
            self.db.clear_database()
            self._update_stats_display()
            self._show_empty_results_placeholder("Photo index has been reset.")

    def _on_upload_query(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")]
        path = filedialog.askopenfilename(title="Select Target Person Photo", filetypes=filetypes)
        if not path:
            return

        self.query_image_path = path
        img = self.engine.read_image(path)
        if img is None:
            messagebox.showerror("Error", "Could not open selected image.")
            return

        self.query_image_cv = img
        faces = self.engine.detect_and_extract_faces(img)
        self.query_faces = faces

        if not faces:
            messagebox.showwarning(
                "No Face Detected", 
                "No human face could be clearly identified in this photo. Please try a clearer picture."
            )
            self.btn_search.configure(state="disabled")
            return

        self.selected_face_idx = 0
        self._render_query_preview()
        self.btn_search.configure(state="normal")

    def _render_query_preview(self):
        if self.query_image_cv is None or not self.query_faces:
            return

        face = self.query_faces[self.selected_face_idx]
        bx, by, bw, bh = face["bbox"]

        # Draw a bounding rectangle on a copy of the image
        preview_cv = self.query_image_cv.copy()
        cv2.rectangle(preview_cv, (bx, by), (bx + bw, by + bh), (0, 230, 64), 3)

        # Crop thumbnail centered around face with margin
        h, w = preview_cv.shape[:2]
        margin = int(max(bw, bh) * 0.3)
        crop_x1 = max(0, bx - margin)
        crop_y1 = max(0, by - margin)
        crop_x2 = min(w, bx + bw + margin)
        crop_y2 = min(h, by + bh + margin)
        face_crop = preview_cv[crop_y1:crop_y2, crop_x1:crop_x2]

        # Convert to RGB PIL Image
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_crop_rgb)
        pil_img.thumbnail((140, 140))

        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        self.query_preview_label.configure(image=ctk_img, text="")
        self.thumbnail_cache["query"] = ctk_img

        # Multi-face selector buttons if more than 1 face in image
        for child in self.face_selector_row.winfo_children():
            child.destroy()

        if len(self.query_faces) > 1:
            self.face_select_info.configure(
                text=f"Detected {len(self.query_faces)} faces. Choose which person to find:"
            )
            for i in range(len(self.query_faces)):
                btn = ctk.CTkButton(
                    self.face_selector_row, 
                    text=f"Face {i+1}", 
                    width=60,
                    height=26,
                    fg_color="#2563eb" if i == self.selected_face_idx else "#334155",
                    command=lambda idx=i: self._select_face(idx)
                )
                btn.pack(side="left", padx=3)
        else:
            self.face_select_info.configure(text="Face detected successfully! Click Search below.")

    def _select_face(self, idx: int):
        self.selected_face_idx = idx
        self._render_query_preview()

    def _on_search(self):
        if not self.query_faces:
            return

        stats = self.db.get_stats()
        if stats["photos"] == 0:
            messagebox.showwarning(
                "No Photos Indexed", 
                "You haven't indexed any photos yet! Please select your photo folder in Step 1 and run indexing first."
            )
            return

        query_face = self.query_faces[self.selected_face_idx]
        query_emb = query_face["embedding"]
        threshold = self.threshold_slider.get()

        # Load all pre-indexed embeddings from SQLite
        embeddings_matrix, meta_list = self.db.load_all_embeddings()
        
        # Fast vector search (takes < 0.05 seconds!)
        matches = self.engine.search_similar(
            query_embedding=query_emb,
            embeddings_matrix=embeddings_matrix,
            metadata_list=meta_list,
            threshold=threshold
        )
        self.search_results = matches
        self._render_search_results()

    def _show_empty_results_placeholder(self, text: str):
        for widget in self.results_scroll.winfo_children():
            widget.destroy()
        placeholder = ctk.CTkLabel(
            self.results_scroll, 
            text=text, 
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        placeholder.pack(pady=40)
        self.results_title.configure(text="Matches Found: 0")

    def _render_search_results(self):
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        self.thumbnail_cache.clear()
        total_matches = len(self.search_results)
        self.results_title.configure(text=f"Matches Found: {total_matches}")

        if total_matches == 0:
            self._show_empty_results_placeholder(
                "No matching photos found at this strictness level.\nTry lowering the 'Search Accuracy' slider slightly."
            )
            return

        # Render list of matching photo cards
        for idx, match in enumerate(self.search_results):
            filepath = match["filepath"]
            score_pct = match["similarity_percent"]

            card = ctk.CTkFrame(self.results_scroll, fg_color="#1e293b", corner_radius=8)
            card.pack(fill="x", padx=5, pady=4)

            # Left thumbnail (load asynchronously or on demand)
            thumb_label = ctk.CTkLabel(card, text="📷", width=70, height=70, fg_color="#0f172a", corner_radius=6)
            thumb_label.pack(side="left", padx=8, pady=8)

            try:
                with Image.open(filepath) as pil_img:
                    pil_img.thumbnail((70, 70))
                    ctk_thumb = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
                    thumb_label.configure(image=ctk_thumb, text="")
                    self.thumbnail_cache[f"thumb_{idx}"] = ctk_thumb
            except Exception:
                pass

            # Center Info: File Name, Folder, Score Badge
            info_frame = ctk.CTkFrame(card, fg_color="transparent")
            info_frame.pack(side="left", fill="both", expand=True, padx=10, pady=8)

            fname = os.path.basename(filepath)
            fname_label = ctk.CTkLabel(
                info_frame, 
                text=fname, 
                font=ctk.CTkFont(size=14, weight="bold"),
                anchor="w"
            )
            fname_label.pack(fill="x", anchor="w")

            fdir = os.path.dirname(filepath)
            dir_label = ctk.CTkLabel(
                info_frame, 
                text=fdir, 
                font=ctk.CTkFont(size=11),
                text_color="gray",
                anchor="w"
            )
            dir_label.pack(fill="x", anchor="w")

            # Match badge color based on confidence
            badge_color = "#22c55e" if score_pct >= 60 else "#eab308"
            score_badge = ctk.CTkLabel(
                info_frame,
                text=f"Match Confidence: {score_pct}%",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=badge_color,
                anchor="w"
            )
            score_badge.pack(fill="x", anchor="w", pady=(2, 0))

            # Right Action Buttons
            btn_frame = ctk.CTkFrame(card, fg_color="transparent")
            btn_frame.pack(side="right", padx=10, pady=8)

            btn_open = ctk.CTkButton(
                btn_frame, 
                text="Open Photo", 
                width=100, 
                height=30,
                command=lambda p=filepath: self._open_photo(p)
            )
            btn_open.pack(pady=(0, 4))

            btn_reveal = ctk.CTkButton(
                btn_frame, 
                text="Show in Folder", 
                width=100, 
                height=30,
                fg_color="#334155",
                hover_color="#475569",
                command=lambda p=filepath: self._reveal_in_folder(p)
            )
            btn_reveal.pack()

    def _open_photo(self, filepath: str):
        try:
            if sys.platform == "win32":
                os.startfile(filepath)
            else:
                subprocess.Popen(["xdg-open", filepath])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open photo:\n{e}")

    def _reveal_in_folder(self, filepath: str):
        try:
            norm_path = os.path.normpath(filepath)
            if sys.platform == "win32":
                subprocess.Popen(f'explorer /select,"{norm_path}"')
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(norm_path)])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reveal folder:\n{e}")

if __name__ == "__main__":
    app = OfflineFaceFinderApp()
    app.mainloop()
