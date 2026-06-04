"""
PCB Defect Inspection HMI - Step 1: Core Layout
================================================
Features in this step:
  - Professional dark-theme CustomTkinter dashboard
  - Dual panel: Live Feed + Frozen Frame (Defect Snapshot)
  - Status banner: PASS (green) / REJECT (red) with defect count
  - Confidence score display per detection
  - Start / Stop / Auto Mode / Freeze+Capture / Save Report buttons
  - Threaded camera capture (non-blocking UI)
  - YOLOv11 real-time inference overlay

Requirements:
  pip install customtkinter opencv-python ultralytics Pillow
"""

import customtkinter as ctk
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
MODEL_PATH  = "runs/detect/train_v2/weights/best.pt"   # adjust to your weights
CAMERA_ID   = 0                                         # 0 = default webcam
CONF_THRESH = 0.35                                      # minimum confidence

# Defect class colours (BGR for OpenCV, then converted for display)
CLASS_COLORS = {
    "missing_comp":  (0,   80, 220),   # blue-ish
    "missing_tb":    (0,  180, 255),   # cyan
    "broken_comp":   (0,   30, 220),   # red (BGR)
    "bent_pin":      (20, 180,  20),   # green
}
DEFAULT_COLOR = (30, 30, 200)

# ── Utility ───────────────────────────────────────────────────────────────────
def bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"

def cv2_frame_to_ctk(frame, width, height):
    """Resize and convert OpenCV BGR frame → CTkImage."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb).resize((width, height), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                        size=(width, height))


# ── Main Application ──────────────────────────────────────────────────────────
class PCBInspectionHMI(ctk.CTk):

    PANEL_W = 560
    PANEL_H = 420

    def __init__(self):
        super().__init__()

        # ── Window setup ──────────────────────────────────────────────────────
        self.title("FYP: PCB Defect Inspection System")
        self.geometry("1280x820")
        self.resizable(True, True)
        self.configure(fg_color="#0d0d12")

        # ── State ─────────────────────────────────────────────────────────────
        self.camera_running  = False
        self.auto_mode       = False
        self.frozen_frame    = None          # last captured defect snapshot
        self.frozen_detections = []          # detections on frozen frame
        self.total_inspected = 0
        self.total_pass      = 0
        self.total_reject    = 0
        self.cap             = None
        self.thread          = None

        # ── Load model ────────────────────────────────────────────────────────
        self.model = None
        self._load_model()

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_header()
        self._build_video_panels()
        self._build_status_bar()
        self._build_controls()
        self._build_stat_cards()

        # ── Start blank panels ────────────────────────────────────────────────
        self._show_placeholder(self.live_label,   "Live Feed")
        self._show_placeholder(self.frozen_label, "Frozen Frame  (Defect Snapshot)")

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = YOLO(MODEL_PATH)
                print(f"[INFO] Model loaded: {MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Could not load model: {e}")
        else:
            print(f"[WARN] Model not found at {MODEL_PATH}. Running in demo mode.")

    # ── UI builders ───────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color="#111118", corner_radius=0, height=60)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr,
            text="⬡  PCB DEFECT INSPECTION SYSTEM",
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

    def _build_video_panels(self):
        panels_frame = ctk.CTkFrame(self, fg_color="#0d0d12")
        panels_frame.pack(fill="x", padx=16, pady=(8, 0))

        # Live panel
        live_card = ctk.CTkFrame(panels_frame, fg_color="#13131e",
                                  corner_radius=12, border_width=1,
                                  border_color="#1e2a3a")
        live_card.pack(side="left", padx=(0, 8), expand=True, fill="both")

        ctk.CTkLabel(live_card, text="LIVE FEED",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#445566").pack(pady=(8, 2))

        self.live_label = ctk.CTkLabel(live_card, text="")
        self.live_label.pack(padx=8, pady=(0, 8))

        # Frozen / snapshot panel
        frozen_card = ctk.CTkFrame(panels_frame, fg_color="#13131e",
                                    corner_radius=12, border_width=1,
                                    border_color="#1e2a3a")
        frozen_card.pack(side="left", padx=(8, 0), expand=True, fill="both")

        ctk.CTkLabel(frozen_card, text="FROZEN FRAME  ·  DEFECT SNAPSHOT",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#445566").pack(pady=(8, 2))

        self.frozen_label = ctk.CTkLabel(frozen_card, text="")
        self.frozen_label.pack(padx=8, pady=(0, 8))

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

    def _build_controls(self):
        ctrl = ctk.CTkFrame(self, fg_color="#0d0d12")
        ctrl.pack(fill="x", padx=16, pady=(10, 0))

        btn_specs = [
            ("▶  Start Camera",  "#1a5c2a", "#22aa44", self.start_camera),
            ("■  Stop Camera",   "#5c1a1a", "#cc3333", self.stop_camera),
            ("⟳  Auto Mode",    "#1a3a5c", "#2277cc", self.toggle_auto),
            ("⊡  Freeze / Capture", "#2a1a5c", "#7744cc", self.freeze_capture),
            ("⬛  Save Report",  "#5c4a0a", "#cc9922", self.save_report),
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

    def _build_stat_cards(self):
        cards_frame = ctk.CTkFrame(self, fg_color="#0d0d12")
        cards_frame.pack(fill="x", padx=16, pady=(10, 12))

        specs = [
            ("TOTAL INSPECTED", "total_val", "#00ccff"),
            ("PASSED",          "pass_val",  "#33cc66"),
            ("REJECTED",        "reject_val","#ee3333"),
            ("INFERENCE (ms)",  "fps_val",   "#aaaaaa"),
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

    # ── Placeholder image ─────────────────────────────────────────────────────
    def _show_placeholder(self, label, caption=""):
        blank = Image.new("RGB", (self.PANEL_W, self.PANEL_H), (20, 22, 30))
        # draw text on it
        import PIL.ImageDraw
        draw = PIL.ImageDraw.Draw(blank)
        draw.text((self.PANEL_W // 2 - 60, self.PANEL_H // 2 - 10),
                  f"[ {caption} ]", fill=(60, 80, 100))
        ctk_img = ctk.CTkImage(light_image=blank, dark_image=blank,
                               size=(self.PANEL_W, self.PANEL_H))
        label.configure(image=ctk_img, text="")
        label._image = ctk_img

    # ── Camera thread ─────────────────────────────────────────────────────────
    def start_camera(self):
        if self.camera_running:
            return
        self.cap = cv2.VideoCapture(CAMERA_ID)
        if not self.cap.isOpened():
            self._set_status("✖  Camera not found — check CAMERA_ID", "#ee3333")
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
        """Manually capture current frame as frozen snapshot."""
        # Handled inside camera loop — flag set here
        self._trigger_freeze = True

    # ── Core inference loop ───────────────────────────────────────────────────
    def _camera_loop(self):
        self._trigger_freeze = False

        while self.camera_running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            annotated, detections = self._run_inference(frame)
            elapsed_ms = round((time.time() - t0) * 1000, 1)

            # Update live panel
            live_img = cv2_frame_to_ctk(annotated, self.PANEL_W, self.PANEL_H)
            self.live_label.configure(image=live_img, text="")
            self.live_label._image = live_img

            # Update status + stats
            defect_count = len(detections)
            self.after(0, self._update_stats, defect_count, elapsed_ms)

            # Freeze if defect found (auto mode) or manual trigger
            if (self.auto_mode and defect_count > 0) or self._trigger_freeze:
                self._trigger_freeze = False
                self.frozen_frame = annotated.copy()
                self.frozen_detections = detections
                self.after(0, self._update_frozen_panel)

            time.sleep(0.01)

    def _run_inference(self, frame):
        """Run YOLOv11 inference and draw boxes. Returns (annotated, detections)."""
        detections = []

        if self.model is None:
            # Demo mode: no model, just return plain frame
            return frame, detections

        results = self.model(frame, stream=True, conf=CONF_THRESH, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                cls_id    = int(box.cls[0])
                cls_name  = self.model.names[cls_id]
                conf      = math.ceil(box.conf[0] * 100) / 100
                color     = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
                label_txt = f"{cls_name.upper()}  {conf}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label background
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

    def _update_stats(self, defect_count, elapsed_ms):
        self.total_inspected += 1
        if defect_count == 0:
            self.total_pass += 1
            self._set_status(
                f"✔  PASS  —  No defects detected",
                "#33cc66", bg="#0a1f0a"
            )
        else:
            self.total_reject += 1
            self._set_status(
                f"✖  REJECT  —  {defect_count} defect{'s' if defect_count > 1 else ''} found!",
                "#ffffff", bg="#4a0808"
            )

        self.total_val.configure( text=str(self.total_inspected))
        self.pass_val.configure(  text=str(self.total_pass))
        self.reject_val.configure(text=str(self.total_reject))
        self.fps_val.configure(   text=str(elapsed_ms))

    def _set_status(self, text, color="#33cc66", bg="#112200"):
        self.status_frame.configure(fg_color=bg)
        self.status_label.configure(text=text, text_color=color)

    # ── Save report placeholder ───────────────────────────────────────────────
    def save_report(self):
        """Placeholder — full implementation added in Step 4."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[INFO] Save report requested at {ts}  —  will be implemented in Step 4")
        self._set_status(f"⬛  Report save coming in Step 4…", "#cc9922", "#2a1a00")

    # ── Close handler ─────────────────────────────────────────────────────────
    def on_closing(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PCBInspectionHMI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()