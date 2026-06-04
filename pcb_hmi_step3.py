"""
PCB Defect Inspection HMI - Step 3: Statistics Chart + Sound Alert
===================================================================
New in this step:
  - Defect class bar chart (matplotlib embedded in tkinter)
  - Chart updates live every time a defect is detected
  - Sound alert (beep) on defect detection using winsound (Windows)
    or os.system fallback for Linux/Mac
  - "Refresh Chart" button to manually redraw
  - Pass vs Reject pie-style summary next to bar chart

All Step 1 + Step 2 features retained.

Requirements:
  pip install customtkinter opencv-python ultralytics Pillow matplotlib
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import cv2
import threading
import time
from PIL import Image
from ultralytics import YOLO
import math
import os
import sys
from datetime import datetime
from collections import defaultdict

# Matplotlib embedded in tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "runs/detect/train_v2/weights/best.pt"
CAMERA_ID   = 0
CONF_THRESH = 0.35

CLASS_COLORS_BGR = {
    "missing_comp": (0,   80, 220),
    "missing_tb":   (0,  180, 255),
    "broken_comp":  (30,  30, 200),
    "bent_pin":     (20, 180,  20),
}
DEFAULT_COLOR_BGR = (30, 30, 200)

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

# Chart colours (matplotlib hex)
CHART_COLORS = {
    "missing_comp": "#5599ff",
    "missing_tb":   "#33ccee",
    "broken_comp":  "#ff4444",
    "bent_pin":     "#44cc66",
}
CHART_BG   = "#0f0f18"
CHART_AXES = "#13131e"

# ── Sound alert ───────────────────────────────────────────────────────────────
def play_alert():
    """Non-blocking beep on defect detection."""
    def _beep():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 300)   # 1000 Hz, 300 ms
            elif sys.platform == "darwin":
                os.system("afplay /System/Library/Sounds/Funk.aiff &")
            else:
                os.system("beep -f 1000 -l 300 2>/dev/null || "
                          "python3 -c \"import sys; sys.stdout.write('\\a'); "
                          "sys.stdout.flush()\"")
        except Exception:
            pass   # silent fail — never crash the UI for a beep
    threading.Thread(target=_beep, daemon=True).start()

# ── Utilities ─────────────────────────────────────────────────────────────────
def cv2_frame_to_ctk(frame, width, height):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb).resize((width, height), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(width, height))


# ── Main Application ──────────────────────────────────────────────────────────
class PCBInspectionHMI(ctk.CTk):

    PANEL_W = 520
    PANEL_H = 370

    def __init__(self):
        super().__init__()
        self.title("FYP: PCB Defect Inspection System")
        self.geometry("1280x950")
        self.resizable(True, True)
        self.configure(fg_color="#0d0d12")

        # State
        self.camera_running  = False
        self.auto_mode       = False
        self.frozen_frame    = None
        self.cap             = None
        self.thread          = None
        self._trigger_freeze = False
        self.total_inspected = 0
        self.total_pass      = 0
        self.total_reject    = 0
        self.log_entry_count = 0
        self._last_beep_time = 0

        # Defect counters for chart
        self.defect_counts = defaultdict(int)

        self.model = None
        self._load_model()

        self._build_header()
        self._build_video_panels()
        self._build_status_bar()
        self._build_controls()
        self._build_stat_cards()

        # Bottom section: log table + chart side by side
        self._build_bottom_section()

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
            print("[WARN] Model not found — demo mode")

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

        for i, (label_text, attr) in enumerate([
            ("LIVE FEED",                         "live_label"),
            ("FROZEN FRAME  ·  DEFECT SNAPSHOT",  "frozen_label"),
        ]):
            card = ctk.CTkFrame(panels_frame, fg_color="#13131e",
                                corner_radius=12, border_width=1,
                                border_color="#1e2a3a")
            card.pack(side="left", padx=(0 if i == 0 else 8,
                                          8 if i == 0 else 0),
                      expand=True, fill="both")
            ctk.CTkLabel(card, text=label_text,
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="#445566").pack(pady=(8, 2))
            lbl = ctk.CTkLabel(card, text="")
            lbl.pack(padx=8, pady=(0, 8))
            setattr(self, attr, lbl)

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        self.status_frame = ctk.CTkFrame(self, corner_radius=10,
                                          height=52, fg_color="#112200")
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
            ("▶  Start Camera",     "#1a5c2a", "#22aa44", self.start_camera),
            ("■  Stop Camera",      "#5c1a1a", "#cc3333", self.stop_camera),
            ("⟳  Auto Mode",       "#1a3a5c", "#2277cc", self.toggle_auto),
            ("⊡  Freeze / Capture", "#2a1a5c", "#7744cc", self.freeze_capture),
            ("⬛  Save Report",     "#5c4a0a", "#cc9922", self.save_report),
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

    # ── Bottom section: log table LEFT + chart RIGHT ───────────────────────────
    def _build_bottom_section(self):
        bottom = ctk.CTkFrame(self, fg_color="#0d0d12")
        bottom.pack(fill="both", padx=16, pady=(10, 12), expand=True)

        self._build_log_table(bottom)   # left half
        self._build_chart(bottom)       # right half

    # ── Log table ─────────────────────────────────────────────────────────────
    def _build_log_table(self, parent):
        log_outer = ctk.CTkFrame(parent, fg_color="#13131e", corner_radius=12,
                                  border_width=1, border_color="#1e2a3a")
        log_outer.pack(side="left", fill="both", expand=True,
                       padx=(0, 6), pady=0)

        hdr_row = ctk.CTkFrame(log_outer, fg_color="#13131e", corner_radius=0)
        hdr_row.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(hdr_row, text="DEFECT INSPECTION LOG",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#445566").pack(side="left")
        self.log_count_label = ctk.CTkLabel(
            hdr_row, text="0 entries",
            font=ctk.CTkFont(size=11), text_color="#334455"
        )
        self.log_count_label.pack(side="left", padx=10)
        ctk.CTkButton(
            hdr_row, text="✕  Clear Log", width=110, height=28,
            fg_color="#2a1515", hover_color="#441515",
            font=ctk.CTkFont(size=11),
            corner_radius=6, command=self._clear_log
        ).pack(side="right")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("PCB.Treeview",
                        background="#0f0f18", foreground="#aabbcc",
                        fieldbackground="#0f0f18", rowheight=28,
                        font=("Courier New", 11), borderwidth=0)
        style.configure("PCB.Treeview.Heading",
                        background="#111120", foreground="#445566",
                        font=("Courier New", 10, "bold"),
                        relief="flat", borderwidth=0)
        style.map("PCB.Treeview",
                  background=[("selected", "#1e3a5f")],
                  foreground=[("selected", "#ffffff")])
        style.layout("PCB.Treeview",
                     [("Treeview.treearea", {"sticky": "nswe"})])

        columns = ("#", "Timestamp", "Defect Class", "Confidence", "Status")
        self.tree = ttk.Treeview(log_outer, columns=columns, show="headings",
                                  style="PCB.Treeview", height=7)
        for col, w in zip(columns, [50, 150, 180, 110, 100]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center", stretch=False)
        for cls, bg in ROW_COLORS.items():
            self.tree.tag_configure(cls, background=bg,
                                    foreground=ROW_FG.get(cls, "#aabbcc"))

        sb = ttk.Scrollbar(log_outer, orient="vertical",
                           command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True,
                       padx=(12, 0), pady=(0, 10))
        sb.pack(side="right", fill="y", padx=(0, 8), pady=(0, 10))

    # ── Chart (NEW in Step 3) ─────────────────────────────────────────────────
    def _build_chart(self, parent):
        chart_outer = ctk.CTkFrame(parent, fg_color="#13131e", corner_radius=12,
                                    border_width=1, border_color="#1e2a3a",
                                    width=380)
        chart_outer.pack(side="left", fill="both", padx=(6, 0), pady=0)
        chart_outer.pack_propagate(False)

        # Chart header
        hdr = ctk.CTkFrame(chart_outer, fg_color="#13131e", corner_radius=0)
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(hdr, text="DEFECT STATISTICS",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#445566").pack(side="left")
        ctk.CTkButton(
            hdr, text="↺  Refresh", width=90, height=28,
            fg_color="#1a2a1a", hover_color="#2a3a2a",
            font=ctk.CTkFont(size=11),
            corner_radius=6, command=self._redraw_chart
        ).pack(side="right")

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(3.6, 3.2))
        self.fig.patch.set_facecolor(CHART_BG)
        self.ax.set_facecolor(CHART_AXES)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_outer)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                          padx=10, pady=(0, 10))
        self._redraw_chart()   # draw empty chart initially

    def _redraw_chart(self):
        """Redraw the defect count bar chart."""
        self.ax.clear()
        self.ax.set_facecolor(CHART_AXES)

        classes = ["missing_comp", "missing_tb", "broken_comp", "bent_pin"]
        labels  = ["Missing\nComp", "Missing\nTB", "Broken\nComp", "Bent\nPin"]
        counts  = [self.defect_counts[c] for c in classes]
        colors  = [CHART_COLORS[c] for c in classes]

        bars = self.ax.bar(labels, counts, color=colors,
                           edgecolor="#0d0d12", linewidth=0.8,
                           width=0.55)

        # Value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                self.ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(count),
                    ha="center", va="bottom",
                    color="#ffffff", fontsize=10, fontweight="bold"
                )

        # Styling
        self.ax.set_title("Defect Count by Class",
                          color="#667788", fontsize=10, pad=8)
        self.ax.set_ylabel("Count", color="#445566", fontsize=9)
        self.ax.tick_params(colors="#445566", labelsize=8)
        self.ax.spines["bottom"].set_color("#1e2a3a")
        self.ax.spines["left"].set_color("#1e2a3a")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True)
        )

        # Pass / Reject mini summary at bottom
        total = self.total_pass + self.total_reject
        if total > 0:
            pass_pct   = round(self.total_pass   / total * 100)
            reject_pct = round(self.total_reject / total * 100)
            summary = f"Pass {pass_pct}%  ·  Reject {reject_pct}%"
        else:
            summary = "No inspections yet"

        self.fig.text(0.5, 0.01, summary, ha="center",
                      color="#445566", fontsize=8)

        self.fig.tight_layout(rect=[0, 0.05, 1, 1])
        self.canvas.draw()

    # ── Log helpers ───────────────────────────────────────────────────────────
    def _append_log(self, detections):
        ts = datetime.now().strftime("%H:%M:%S")
        if not detections:
            self.log_entry_count += 1
            self.tree.insert("", 0,
                             values=(self.log_entry_count, ts,
                                     "—", "—", "✔ PASS"),
                             tags=("PASS",))
        else:
            for det in detections:
                self.log_entry_count += 1
                cls  = det["class"]
                conf = f"{det['conf']:.2f}"
                tag  = cls if cls in ROW_COLORS else "missing_comp"
                self.tree.insert("", 0,
                                 values=(self.log_entry_count, ts,
                                         cls.upper().replace("_", " "),
                                         conf, "✖ REJECT"),
                                 tags=(tag,))
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
        PIL.ImageDraw.Draw(blank).text(
            (self.PANEL_W // 2 - 60, self.PANEL_H // 2 - 10),
            f"[ {caption} ]", fill=(60, 80, 100)
        )
        ctk_img = ctk.CTkImage(light_image=blank, dark_image=blank,
                               size=(self.PANEL_W, self.PANEL_H))
        label.configure(image=ctk_img, text="")
        label._image = ctk_img

    # ── Camera controls ───────────────────────────────────────────────────────
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

    # ── Camera loop ───────────────────────────────────────────────────────────
    def _camera_loop(self):
        self._trigger_freeze = False
        while self.camera_running:
            t0  = time.time()
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
                color    = CLASS_COLORS_BGR.get(cls_name, DEFAULT_COLOR_BGR)
                label_txt = f"{cls_name.upper()}  {conf}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(
                    label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8),
                              (x1 + tw + 6, y1), color, -1)
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
            self._set_status("✔  PASS  —  No defects detected",
                             "#33cc66", "#0a1f0a")
        else:
            self.total_reject += 1
            self._set_status(
                f"✖  REJECT  —  {defect_count} defect"
                f"{'s' if defect_count > 1 else ''} found!",
                "#ffffff", "#4a0808"
            )
            # Update defect class counters for chart
            for det in detections:
                self.defect_counts[det["class"]] += 1

            # Sound alert — max once every 2 seconds to avoid spam
            now = time.time()
            if now - self._last_beep_time > 2.0:
                self._last_beep_time = now
                play_alert()

            # Refresh chart on every reject
            self.after(0, self._redraw_chart)

        self.total_val.configure( text=str(self.total_inspected))
        self.pass_val.configure(  text=str(self.total_pass))
        self.reject_val.configure(text=str(self.total_reject))
        self.fps_val.configure(   text=str(elapsed_ms))

        # Log entry (every 30 frames for PASS, every defect for REJECT)
        if self.total_inspected % 30 == 0 or defect_count > 0:
            self.after(0, self._append_log, detections)

    def _set_status(self, text, color="#33cc66", bg="#112200"):
        self.status_frame.configure(fg_color=bg)
        self.status_label.configure(text=text, text_color=color)

    # ── Save report placeholder ───────────────────────────────────────────────
    def save_report(self):
        self._set_status("⬛  Report save coming in Step 4…", "#cc9922", "#2a1a00")

    # ── Close ─────────────────────────────────────────────────────────────────
    def on_closing(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        plt.close("all")
        self.destroy()


if __name__ == "__main__":
    app = PCBInspectionHMI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()