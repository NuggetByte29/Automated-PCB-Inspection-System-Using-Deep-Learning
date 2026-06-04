"""
PCB Defect Inspection HMI - Step 2: Defect History Log Table
=============================================================
New in this step:
  - Scrollable defect history table (Treeview via ttk, dark-styled)
  - Columns: #  |  Timestamp  |  Defect Class  |  Confidence  |  Status
  - Color-coded rows per defect type
  - "Clear Log" button
  - Entry count badge next to table header
  - Each REJECT inspection appends ALL detected defects to the log

All Step 1 features retained.

Requirements:
  pip install customtkinter opencv-python ultralytics Pillow
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import cv2
import threading
import time
from PIL import Image, ImageTk
from ultralytics import YOLO
import math
import os
from datetime import datetime

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "runs/detect/train_v2/weights/best.pt"
CAMERA_ID   = 0
CONF_THRESH = 0.35

CLASS_COLORS = {
    "missing_comp": (0,   80, 220),
    "missing_tb":   (0,  180, 255),
    "broken_comp":  (30,  30, 200),
    "bent_pin":     (20, 180,  20),
}
DEFAULT_COLOR = (30, 30, 200)

# Row highlight colours for the log table (hex, dark-mode friendly)
ROW_COLORS = {
    "missing_comp": "#1a2040",
    "missing_tb":   "#0e2a30",
    "broken_comp":  "#2a0e0e",
    "bent_pin":     "#0e2a12",
    "PASS":         "#0d1f0d",
}
ROW_FG = {
    "missing_comp": "#5599ff",
    "missing_tb":   "#33ccee",
    "broken_comp":  "#ff4444",
    "bent_pin":     "#44cc66",
    "PASS":         "#33cc66",
}

# ── Utilities ─────────────────────────────────────────────────────────────────
def cv2_frame_to_ctk(frame, width, height):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb).resize((width, height), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(width, height))


# ── Main Application ──────────────────────────────────────────────────────────
class PCBInspectionHMI(ctk.CTk):

    PANEL_W = 540
    PANEL_H = 390

    def __init__(self):
        super().__init__()
        self.title("FYP: PCB Defect Inspection System")
        self.geometry("1280x900")
        self.resizable(True, True)
        self.configure(fg_color="#0d0d12")

        # State
        self.camera_running    = False
        self.auto_mode         = False
        self.frozen_frame      = None
        self.cap               = None
        self.thread            = None
        self._trigger_freeze   = False
        self.total_inspected   = 0
        self.total_pass        = 0
        self.total_reject      = 0
        self.log_entry_count   = 0

        self.model = None
        self._load_model()

        self._build_header()
        self._build_video_panels()
        self._build_status_bar()
        self._build_controls()
        self._build_stat_cards()
        self._build_log_table()          # ← NEW in Step 2

        self._show_placeholder(self.live_label,   "Live Feed")
        self._show_placeholder(self.frozen_label, "Frozen Frame  ·  Defect Snapshot")

    # ── Model ─────────────────────────────────────────────────────────────────
    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = YOLO(MODEL_PATH)
                print(f"[INFO] Model loaded: {MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Could not load model: {e}")
        else:
            print(f"[WARN] Model not found — demo mode")

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color="#111118", corner_radius=0, height=60)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        ctk.CTkLabel(
            hdr, text="⬡  PCB DEFECT INSPECTION SYSTEM",
            font=ctk.CTkFont(family="Courier New", size=18, weight="bold"),
            text_color="#00ccff"
        ).pack(side="left", padx=24, pady=14)
        self.clock_label = ctk.CTkLabel(
            hdr, text="", font=ctk.CTkFont(size=13), text_color="#667788"
        )
        self.clock_label.pack(side="right", padx=24)
        self._tick_clock()

    def _tick_clock(self):
        self.clock_label.configure(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ── Video panels ──────────────────────────────────────────────────────────
    def _build_video_panels(self):
        panels_frame = ctk.CTkFrame(self, fg_color="#0d0d12")
        panels_frame.pack(fill="x", padx=16, pady=(8, 0))

        for label_text, attr in [
            ("LIVE FEED",                              "live_label"),
            ("FROZEN FRAME  ·  DEFECT SNAPSHOT",       "frozen_label"),
        ]:
            card = ctk.CTkFrame(panels_frame, fg_color="#13131e",
                                corner_radius=12, border_width=1,
                                border_color="#1e2a3a")
            card.pack(side="left", padx=(0 if attr == "live_label" else 8,
                                          8 if attr == "live_label" else 0),
                      expand=True, fill="both")
            ctk.CTkLabel(card, text=label_text,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="#445566").pack(pady=(8, 2))
            lbl = ctk.CTkLabel(card, text="")
            lbl.pack(padx=8, pady=(0, 8))
            setattr(self, attr, lbl)

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        self.status_frame = ctk.CTkFrame(self, corner_radius=10, height=52,
                                          fg_color="#112200")
        self.status_frame.pack(fill="x", padx=16, pady=(8, 0))
        self.status_frame.pack_propagate(False)
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="●  SYSTEM READY  —  Start camera to begin inspection",
            font=ctk.CTkFont(family="Courier New", size=15, weight="bold"),
            text_color="#33cc66"
        )
        self.status_label.pack(expand=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    def _build_controls(self):
        ctrl = ctk.CTkFrame(self, fg_color="#0d0d12")
        ctrl.pack(fill="x", padx=16, pady=(10, 0))

        btn_specs = [
            ("▶  Start Camera",    "#1a5c2a", "#22aa44", self.start_camera),
            ("■  Stop Camera",     "#5c1a1a", "#cc3333", self.stop_camera),
            ("⟳  Auto Mode",      "#1a3a5c", "#2277cc", self.toggle_auto),
            ("⊡  Freeze / Capture","#2a1a5c", "#7744cc", self.freeze_capture),
            ("⬛  Save Report",    "#5c4a0a", "#cc9922", self.save_report),
        ]
        self.auto_btn = None
        for label, fg, hover, cmd in btn_specs:
            btn = ctk.CTkButton(
                ctrl, text=label, width=190, height=44,
                fg_color=fg, hover_color=hover,
                font=ctk.CTkFont(size=13, weight="bold"),
                corner_radius=8, command=cmd
            )
            btn.pack(side="left", padx=6)
            if "Auto" in label:
                self.auto_btn = btn

    # ── Stat cards ────────────────────────────────────────────────────────────
    def _build_stat_cards(self):
        cards_frame = ctk.CTkFrame(self, fg_color="#0d0d12")
        cards_frame.pack(fill="x", padx=16, pady=(10, 0))

        specs = [
            ("TOTAL INSPECTED", "total_val",  "#00ccff"),
            ("PASSED",          "pass_val",   "#33cc66"),
            ("REJECTED",        "reject_val", "#ee3333"),
            ("INFERENCE (ms)",  "fps_val",    "#aaaaaa"),
        ]
        for title, attr, color in specs:
            card = ctk.CTkFrame(cards_frame, fg_color="#13131e",
                                corner_radius=10, border_width=1,
                                border_color="#1e2a3a", width=200, height=72)
            card.pack(side="left", padx=6, expand=True, fill="x")
            card.pack_propagate(False)
            ctk.CTkLabel(card, text=title,
                         font=ctk.CTkFont(size=10, weight="bold"),
                         text_color="#445566").pack(pady=(10, 0))
            lbl = ctk.CTkLabel(card, text="0",
                               font=ctk.CTkFont(size=26, weight="bold"),
                               text_color=color)
            lbl.pack()
            setattr(self, attr, lbl)

    # ── Log table (NEW in Step 2) ──────────────────────────────────────────────
    def _build_log_table(self):
        # Container frame
        log_outer = ctk.CTkFrame(self, fg_color="#13131e", corner_radius=12,
                                  border_width=1, border_color="#1e2a3a")
        log_outer.pack(fill="both", padx=16, pady=(10, 12), expand=True)

        # Table header row
        hdr_row = ctk.CTkFrame(log_outer, fg_color="#13131e", corner_radius=0)
        hdr_row.pack(fill="x", padx=12, pady=(10, 6))

        ctk.CTkLabel(hdr_row,
                     text="DEFECT INSPECTION LOG",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#445566").pack(side="left")

        self.log_count_label = ctk.CTkLabel(
            hdr_row, text="0 entries",
            font=ctk.CTkFont(size=11),
            text_color="#334455"
        )
        self.log_count_label.pack(side="left", padx=10)

        clear_btn = ctk.CTkButton(
            hdr_row, text="✕  Clear Log", width=110, height=28,
            fg_color="#2a1515", hover_color="#441515",
            font=ctk.CTkFont(size=11),
            corner_radius=6, command=self._clear_log
        )
        clear_btn.pack(side="right")

        # Style the ttk Treeview to match dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("PCB.Treeview",
                        background="#0f0f18",
                        foreground="#aabbcc",
                        fieldbackground="#0f0f18",
                        rowheight=28,
                        font=("Courier New", 11),
                        borderwidth=0)
        style.configure("PCB.Treeview.Heading",
                        background="#111120",
                        foreground="#445566",
                        font=("Courier New", 10, "bold"),
                        relief="flat",
                        borderwidth=0)
        style.map("PCB.Treeview",
                  background=[("selected", "#1e3a5f")],
                  foreground=[("selected", "#ffffff")])
        style.layout("PCB.Treeview", [
            ("Treeview.treearea", {"sticky": "nswe"})
        ])

        # Treeview widget
        columns = ("#", "Timestamp", "Defect Class", "Confidence", "Status")
        self.tree = ttk.Treeview(
            log_outer, columns=columns, show="headings",
            style="PCB.Treeview", height=6
        )

        col_widths = [50, 175, 200, 120, 100]
        for col, width in zip(columns, col_widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="center", stretch=False)

        # Tag colours per defect class
        for cls, bg in ROW_COLORS.items():
            self.tree.tag_configure(cls, background=bg,
                                    foreground=ROW_FG.get(cls, "#aabbcc"))

        # Scrollbar
        scrollbar = ttk.Scrollbar(log_outer, orient="vertical",
                                   command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 10))
        scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=(0, 10))

    # ── Log helpers ───────────────────────────────────────────────────────────
    def _append_log(self, detections):
        """Add one row per detection to the log table."""
        ts = datetime.now().strftime("%H:%M:%S")

        if not detections:
            # PASS row
            self.log_entry_count += 1
            self.tree.insert(
                "", 0,
                values=(self.log_entry_count, ts, "—", "—", "✔ PASS"),
                tags=("PASS",)
            )
        else:
            for det in detections:
                self.log_entry_count += 1
                cls   = det["class"]
                conf  = f"{det['conf']:.2f}"
                tag   = cls if cls in ROW_COLORS else "missing_comp"
                self.tree.insert(
                    "", 0,
                    values=(self.log_entry_count, ts,
                            cls.upper().replace("_", " "), conf, "✖ REJECT"),
                    tags=(tag,)
                )

        self.log_count_label.configure(text=f"{self.log_entry_count} entries")

    def _clear_log(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.log_entry_count = 0
        self.log_count_label.configure(text="0 entries")

    # ── Placeholder ───────────────────────────────────────────────────────────
    def _show_placeholder(self, label, caption=""):
        import PIL.ImageDraw
        blank = Image.new("RGB", (self.PANEL_W, self.PANEL_H), (20, 22, 30))
        draw  = PIL.ImageDraw.Draw(blank)
        draw.text((self.PANEL_W // 2 - 60, self.PANEL_H // 2 - 10),
                  f"[ {caption} ]", fill=(60, 80, 100))
        ctk_img = ctk.CTkImage(light_image=blank, dark_image=blank,
                               size=(self.PANEL_W, self.PANEL_H))
        label.configure(image=ctk_img, text="")
        label._image = ctk_img

    # ── Camera ────────────────────────────────────────────────────────────────
    def start_camera(self):
        if self.camera_running:
            return
        self.cap = cv2.VideoCapture(CAMERA_ID)
        if not self.cap.isOpened():
            self._set_status("✖  Camera not found", "#ee3333")
            return
        self.camera_running = True
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()
        self._set_status("●  Camera running  —  Inspecting…", "#33cc66")

    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._show_placeholder(self.live_label, "Live Feed")
        self._set_status("●  Camera stopped", "#888888")
        self.auto_mode = False
        if self.auto_btn:
            self.auto_btn.configure(fg_color="#1a3a5c", text="⟳  Auto Mode")

    def toggle_auto(self):
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self.auto_btn.configure(fg_color="#2a6622", text="⟳  Auto ON")
        else:
            self.auto_btn.configure(fg_color="#1a3a5c", text="⟳  Auto Mode")

    def freeze_capture(self):
        self._trigger_freeze = True

    def _camera_loop(self):
        self._trigger_freeze = False
        while self.camera_running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            annotated, detections = self._run_inference(frame)
            elapsed_ms = round((time.time() - t0) * 1000, 1)

            live_img = cv2_frame_to_ctk(annotated, self.PANEL_W, self.PANEL_H)
            self.live_label.configure(image=live_img, text="")
            self.live_label._image = live_img

            defect_count = len(detections)
            self.after(0, self._update_stats, defect_count, elapsed_ms, detections)

            if (self.auto_mode and defect_count > 0) or self._trigger_freeze:
                self._trigger_freeze = False
                self.frozen_frame = annotated.copy()
                self.after(0, self._update_frozen_panel)

            time.sleep(0.01)

    def _run_inference(self, frame):
        detections = []
        if self.model is None:
            return frame, detections
        results = self.model(frame, stream=True, conf=CONF_THRESH, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                cls_id   = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf     = math.ceil(box.conf[0] * 100) / 100
                color    = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
                label_txt = f"{cls_name.upper()}  {conf}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label_txt, (x1 + 3, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                detections.append({"class": cls_name, "conf": conf,
                                   "bbox": (x1, y1, x2, y2)})
        return frame, detections

    def _update_frozen_panel(self):
        if self.frozen_frame is None:
            return
        frz_img = cv2_frame_to_ctk(self.frozen_frame, self.PANEL_W, self.PANEL_H)
        self.frozen_label.configure(image=frz_img, text="")
        self.frozen_label._image = frz_img

    def _update_stats(self, defect_count, elapsed_ms, detections):
        self.total_inspected += 1
        if defect_count == 0:
            self.total_pass += 1
            self._set_status("✔  PASS  —  No defects detected", "#33cc66", "#0a1f0a")
        else:
            self.total_reject += 1
            n = defect_count
            self._set_status(
                f"✖  REJECT  —  {n} defect{'s' if n > 1 else ''} found!",
                "#ffffff", "#4a0808"
            )

        self.total_val.configure( text=str(self.total_inspected))
        self.pass_val.configure(  text=str(self.total_pass))
        self.reject_val.configure(text=str(self.total_reject))
        self.fps_val.configure(   text=str(elapsed_ms))

        # Append to log every N frames to avoid flooding (every 30 frames ≈ 1s)
        if self.total_inspected % 30 == 0 or defect_count > 0:
            self.after(0, self._append_log, detections)

    def _set_status(self, text, color="#33cc66", bg="#112200"):
        self.status_frame.configure(fg_color=bg)
        self.status_label.configure(text=text, text_color=color)

    def save_report(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[INFO] Save report — Step 4")
        self._set_status("⬛  Report save coming in Step 4…", "#cc9922", "#2a1a00")

    def on_closing(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = PCBInspectionHMI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()