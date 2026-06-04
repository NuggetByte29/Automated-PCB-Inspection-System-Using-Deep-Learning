"""
PCB Defect Inspection HMI - Step 4: ESP32 Serial Control + Save Report
=======================================================================
New in this step:
  - ESP32 serial communication panel (port selector + connect/disconnect)
  - Send START / STOP commands to conveyor via serial
  - Auto-send STOP when defect detected, RESUME after operator clears
  - Save Report as CSV  — full inspection log exported
  - Save Report as PDF  — professional report with snapshot image,
                          summary stats, defect table and chart

All Step 1 + Step 2 + Step 3 features retained.

Requirements:
  pip install customtkinter opencv-python ultralytics Pillow matplotlib
  pip install pyserial reportlab
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import time
from PIL import Image
from ultralytics import YOLO
import math
import os
import sys
import csv
import io
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Serial (optional — graceful if not installed)
try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False
    print("[WARN] pyserial not installed — ESP32 control disabled")
    print("       Run: pip install pyserial")

# PDF (optional — graceful if not installed)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer, Image as RLImage,
                                    HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_OK = True
except ImportError:
    PDF_OK = False
    print("[WARN] reportlab not installed — PDF export disabled")
    print("       Run: pip install reportlab")

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "runs/detect/train_v2/weights/best.pt"
CAMERA_ID    = 0
CONF_THRESH  = 0.35
BAUD_RATE    = 115200

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
CHART_COLORS = {
    "missing_comp": "#5599ff",
    "missing_tb":   "#33ccee",
    "broken_comp":  "#ff4444",
    "bent_pin":     "#44cc66",
}
CHART_BG   = "#0f0f18"
CHART_AXES = "#13131e"

# ── Sound ─────────────────────────────────────────────────────────────────────
def play_alert():
    def _beep():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 300)
            elif sys.platform == "darwin":
                os.system("afplay /System/Library/Sounds/Funk.aiff &")
            else:
                os.system("beep -f 1000 -l 300 2>/dev/null || "
                          "python3 -c \"import sys; sys.stdout.write('\\a');"
                          "sys.stdout.flush()\"")
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()

# ── Utilities ─────────────────────────────────────────────────────────────────
def cv2_frame_to_ctk(frame, width, height):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb).resize((width, height), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                        size=(width, height))


# ── Main Application ──────────────────────────────────────────────────────────
class PCBInspectionHMI(ctk.CTk):

    PANEL_W = 510
    PANEL_H = 360

    def __init__(self):
        super().__init__()
        self.title("FYP: PCB Defect Inspection System")
        self.geometry("1280x980")
        self.resizable(True, True)
        self.configure(fg_color="#0d0d12")

        # ── State ─────────────────────────────────────────────────────────────
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
        self.defect_counts   = defaultdict(int)
        self.session_start   = datetime.now()

        # Full log for export
        self.full_log = []   # list of dicts

        # ESP32 serial
        self.serial_conn  = None
        self.serial_port  = None

        self.model = None
        self._load_model()

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_header()
        self._build_video_panels()
        self._build_status_bar()
        self._build_controls()
        self._build_stat_cards()
        self._build_bottom_section()   # log + chart + esp32 panel

        self._show_placeholder(self.live_label,   "Live Feed")
        self._show_placeholder(self.frozen_label, "Frozen Frame  ·  Defect Snapshot")

    # ── Model ─────────────────────────────────────────────────────────────────
    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = YOLO(MODEL_PATH)
                print(f"[INFO] Model loaded: {MODEL_PATH}")
            except Exception as e:
                print(f"[WARN] Model load failed: {e}")
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
        self.clock_label.configure(
            text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ── Video panels ──────────────────────────────────────────────────────────
    def _build_video_panels(self):
        pf = ctk.CTkFrame(self, fg_color="#0d0d12")
        pf.pack(fill="x", padx=16, pady=(8, 0))
        for i, (txt, attr) in enumerate([
            ("LIVE FEED",                        "live_label"),
            ("FROZEN FRAME  ·  DEFECT SNAPSHOT", "frozen_label"),
        ]):
            card = ctk.CTkFrame(pf, fg_color="#13131e", corner_radius=12,
                                border_width=1, border_color="#1e2a3a")
            card.pack(side="left",
                      padx=(0 if i == 0 else 8, 8 if i == 0 else 0),
                      expand=True, fill="both")
            ctk.CTkLabel(card, text=txt,
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
        cf = ctk.CTkFrame(self, fg_color="#0d0d12")
        cf.pack(fill="x", padx=16, pady=(10, 0))
        for title, attr, color in [
            ("TOTAL INSPECTED", "total_val",  "#00ccff"),
            ("PASSED",          "pass_val",   "#33cc66"),
            ("REJECTED",        "reject_val", "#ee3333"),
            ("INFERENCE (ms)",  "fps_val",    "#aaaaaa"),
        ]:
            card = ctk.CTkFrame(cf, fg_color="#13131e", corner_radius=10,
                                border_width=1, border_color="#1e2a3a",
                                width=200, height=72)
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

    # ── Bottom section ────────────────────────────────────────────────────────
    def _build_bottom_section(self):
        bottom = ctk.CTkFrame(self, fg_color="#0d0d12")
        bottom.pack(fill="both", padx=16, pady=(10, 12), expand=True)

        self._build_log_table(bottom)   # left
        self._build_right_panel(bottom) # right: chart + ESP32

    # ── Log table ─────────────────────────────────────────────────────────────
    def _build_log_table(self, parent):
        outer = ctk.CTkFrame(parent, fg_color="#13131e", corner_radius=12,
                              border_width=1, border_color="#1e2a3a")
        outer.pack(side="left", fill="both", expand=True, padx=(0, 6))

        hdr = ctk.CTkFrame(outer, fg_color="#13131e", corner_radius=0)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(hdr, text="DEFECT INSPECTION LOG",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#445566").pack(side="left")
        self.log_count_label = ctk.CTkLabel(
            hdr, text="0 entries",
            font=ctk.CTkFont(size=11), text_color="#334455")
        self.log_count_label.pack(side="left", padx=10)
        ctk.CTkButton(hdr, text="✕  Clear Log", width=110, height=28,
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

        cols = ("#", "Timestamp", "Defect Class", "Confidence", "Status")
        self.tree = ttk.Treeview(outer, columns=cols, show="headings",
                                  style="PCB.Treeview", height=7)
        for col, w in zip(cols, [50, 150, 180, 110, 100]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center", stretch=False)
        for cls, bg in ROW_COLORS.items():
            self.tree.tag_configure(cls, background=bg,
                                    foreground=ROW_FG.get(cls, "#aabbcc"))
        sb = ttk.Scrollbar(outer, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True,
                       padx=(12, 0), pady=(0, 10))
        sb.pack(side="right", fill="y", padx=(0, 8), pady=(0, 10))

    # ── Right panel: chart on top, ESP32 on bottom ────────────────────────────
    def _build_right_panel(self, parent):
        right = ctk.CTkFrame(parent, fg_color="#0d0d12", width=400)
        right.pack(side="left", fill="both", padx=(6, 0))
        right.pack_propagate(False)

        self._build_chart(right)
        self._build_esp32_panel(right)

    # ── Chart ─────────────────────────────────────────────────────────────────
    def _build_chart(self, parent):
        outer = ctk.CTkFrame(parent, fg_color="#13131e", corner_radius=12,
                              border_width=1, border_color="#1e2a3a")
        outer.pack(fill="both", expand=True, pady=(0, 6))

        hdr = ctk.CTkFrame(outer, fg_color="#13131e", corner_radius=0)
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(hdr, text="DEFECT STATISTICS",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#445566").pack(side="left")
        ctk.CTkButton(hdr, text="↺  Refresh", width=90, height=28,
                      fg_color="#1a2a1a", hover_color="#2a3a2a",
                      font=ctk.CTkFont(size=11),
                      corner_radius=6, command=self._redraw_chart
                      ).pack(side="right")

        self.fig, self.ax = plt.subplots(figsize=(3.6, 2.6))
        self.fig.patch.set_facecolor(CHART_BG)
        self.ax.set_facecolor(CHART_AXES)
        self.canvas = FigureCanvasTkAgg(self.fig, master=outer)
        self.canvas.get_tk_widget().pack(fill="both", expand=True,
                                          padx=10, pady=(0, 10))
        self._redraw_chart()

    def _redraw_chart(self):
        self.ax.clear()
        self.ax.set_facecolor(CHART_AXES)
        classes = ["missing_comp", "missing_tb", "broken_comp", "bent_pin"]
        labels  = ["Missing\nComp", "Missing\nTB", "Broken\nComp", "Bent\nPin"]
        counts  = [self.defect_counts[c] for c in classes]
        colors  = [CHART_COLORS[c] for c in classes]
        bars = self.ax.bar(labels, counts, color=colors,
                           edgecolor="#0d0d12", linewidth=0.8, width=0.55)
        for bar, count in zip(bars, counts):
            if count > 0:
                self.ax.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.1, str(count),
                             ha="center", va="bottom",
                             color="#ffffff", fontsize=9, fontweight="bold")
        self.ax.set_title("Defect Count by Class",
                          color="#667788", fontsize=9, pad=6)
        self.ax.set_ylabel("Count", color="#445566", fontsize=8)
        self.ax.tick_params(colors="#445566", labelsize=7)
        for spine in ["bottom", "left"]:
            self.ax.spines[spine].set_color("#1e2a3a")
        for spine in ["top", "right"]:
            self.ax.spines[spine].set_visible(False)
        self.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        total = self.total_pass + self.total_reject
        summary = (f"Pass {round(self.total_pass/total*100)}%  ·  "
                   f"Reject {round(self.total_reject/total*100)}%"
                   if total > 0 else "No inspections yet")
        self.fig.text(0.5, 0.01, summary, ha="center",
                      color="#445566", fontsize=7)
        self.fig.tight_layout(rect=[0, 0.05, 1, 1])
        self.canvas.draw()

    # ── ESP32 panel (NEW in Step 4) ───────────────────────────────────────────
    def _build_esp32_panel(self, parent):
        outer = ctk.CTkFrame(parent, fg_color="#13131e", corner_radius=12,
                              border_width=1, border_color="#1e2a3a")
        outer.pack(fill="x", pady=(0, 0))

        ctk.CTkLabel(outer, text="ESP32  ·  CONVEYOR CONTROL",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#445566").pack(anchor="w", padx=12, pady=(10, 6))

        # Port row
        port_row = ctk.CTkFrame(outer, fg_color="#13131e")
        port_row.pack(fill="x", padx=12, pady=(0, 6))

        ctk.CTkLabel(port_row, text="Port:",
                     font=ctk.CTkFont(size=12),
                     text_color="#667788").pack(side="left")

        # Dropdown of available serial ports
        ports = self._get_serial_ports()
        self.port_var = tk.StringVar(value=ports[0] if ports else "No ports found")
        self.port_menu = ctk.CTkOptionMenu(
            port_row, variable=self.port_var,
            values=ports if ports else ["No ports found"],
            width=160, height=32,
            fg_color="#1a1a2a", button_color="#2a2a3a",
            font=ctk.CTkFont(size=12)
        )
        self.port_menu.pack(side="left", padx=(8, 0))

        ctk.CTkButton(
            port_row, text="↺", width=32, height=32,
            fg_color="#1a2a1a", hover_color="#2a3a2a",
            font=ctk.CTkFont(size=14),
            corner_radius=6, command=self._refresh_ports
        ).pack(side="left", padx=(6, 0))

        # Connect status indicator
        self.esp_status_dot = ctk.CTkLabel(
            port_row, text="●", font=ctk.CTkFont(size=16),
            text_color="#443333"
        )
        self.esp_status_dot.pack(side="right", padx=(0, 4))
        self.esp_status_lbl = ctk.CTkLabel(
            port_row, text="Disconnected",
            font=ctk.CTkFont(size=11), text_color="#664444"
        )
        self.esp_status_lbl.pack(side="right")

        # Button row
        btn_row = ctk.CTkFrame(outer, fg_color="#13131e")
        btn_row.pack(fill="x", padx=12, pady=(0, 12))

        self.connect_btn = ctk.CTkButton(
            btn_row, text="Connect", width=100, height=34,
            fg_color="#1a3a1a", hover_color="#2a5a2a",
            font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=6, command=self._toggle_esp32_connection
        )
        self.connect_btn.pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="▶  Conveyor ON", width=140, height=34,
            fg_color="#1a3a1a", hover_color="#22aa44",
            font=ctk.CTkFont(size=12),
            corner_radius=6, command=lambda: self._send_serial("START\n")
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="■  Conveyor OFF", width=140, height=34,
            fg_color="#3a1a1a", hover_color="#cc3333",
            font=ctk.CTkFont(size=12),
            corner_radius=6, command=lambda: self._send_serial("STOP\n")
        ).pack(side="left")

    # ── Serial helpers ────────────────────────────────────────────────────────
    def _get_serial_ports(self):
        if not SERIAL_OK:
            return ["pyserial not installed"]
        ports = [p.device for p in serial.tools.list_ports.comports()]
        return ports if ports else ["No ports found"]

    def _refresh_ports(self):
        ports = self._get_serial_ports()
        self.port_menu.configure(values=ports)
        self.port_var.set(ports[0] if ports else "No ports found")

    def _toggle_esp32_connection(self):
        if self.serial_conn and self.serial_conn.is_open:
            self._disconnect_esp32()
        else:
            self._connect_esp32()

    def _connect_esp32(self):
        if not SERIAL_OK:
            messagebox.showerror("Error", "pyserial not installed.\n"
                                          "Run: pip install pyserial")
            return
        port = self.port_var.get()
        if "No ports" in port or "not installed" in port:
            messagebox.showwarning("Warning", "No valid port selected.")
            return
        try:
            self.serial_conn = serial.Serial(port, BAUD_RATE, timeout=1)
            self.serial_port = port
            self.esp_status_dot.configure(text_color="#33cc66")
            self.esp_status_lbl.configure(text=f"Connected  {port}",
                                          text_color="#33cc66")
            self.connect_btn.configure(text="Disconnect", fg_color="#3a1a1a")
            self._set_status(f"●  ESP32 connected on {port}", "#33cc66")
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    def _disconnect_esp32(self):
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.serial_conn = None
        self.serial_port = None
        self.esp_status_dot.configure(text_color="#443333")
        self.esp_status_lbl.configure(text="Disconnected", text_color="#664444")
        self.connect_btn.configure(text="Connect", fg_color="#1a3a1a")

    def _send_serial(self, command: str):
        if not self.serial_conn or not self.serial_conn.is_open:
            self._set_status("✖  ESP32 not connected", "#ee3333", "#2a0a0a")
            return
        try:
            self.serial_conn.write(command.encode())
            print(f"[SERIAL] Sent: {command.strip()}")
        except Exception as e:
            print(f"[SERIAL] Error: {e}")

    # ── Report saving (NEW in Step 4) ─────────────────────────────────────────
    def save_report(self):
        """Show a dialog to choose CSV or PDF."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Save Report")
        dialog.geometry("320x180")
        dialog.resizable(False, False)
        dialog.configure(fg_color="#13131e")
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Choose report format:",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#aabbcc").pack(pady=(20, 12))

        btn_frame = ctk.CTkFrame(dialog, fg_color="#13131e")
        btn_frame.pack()

        ctk.CTkButton(
            btn_frame, text="📄  Save as CSV", width=130, height=40,
            fg_color="#1a3a1a", hover_color="#22aa44",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            command=lambda: [dialog.destroy(), self._save_csv()]
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_frame, text="📑  Save as PDF", width=130, height=40,
            fg_color="#1a2a4a", hover_color="#2255aa",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            command=lambda: [dialog.destroy(), self._save_pdf()]
        ).pack(side="left", padx=8)

        ctk.CTkButton(dialog, text="Cancel", width=80, height=30,
                      fg_color="#2a1515", hover_color="#441515",
                      font=ctk.CTkFont(size=11),
                      corner_radius=6,
                      command=dialog.destroy).pack(pady=(12, 0))

    # ── CSV export ────────────────────────────────────────────────────────────
    def _save_csv(self):
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"pcb_report_{ts}.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["PCB Defect Inspection Report"])
                writer.writerow(["Session Start", self.session_start.strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow(["Generated",     datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow([])
                writer.writerow(["SUMMARY"])
                writer.writerow(["Total Inspected", self.total_inspected])
                writer.writerow(["Passed",          self.total_pass])
                writer.writerow(["Rejected",        self.total_reject])
                writer.writerow([])
                writer.writerow(["DEFECT CLASS COUNTS"])
                for cls, count in self.defect_counts.items():
                    writer.writerow([cls, count])
                writer.writerow([])
                writer.writerow(["INSPECTION LOG"])
                writer.writerow(["#", "Timestamp", "Defect Class",
                                  "Confidence", "Status"])
                for entry in self.full_log:
                    writer.writerow([
                        entry["id"], entry["timestamp"],
                        entry["class"], entry["conf"], entry["status"]
                    ])
            self._set_status(f"✔  CSV saved: {os.path.basename(path)}",
                             "#33cc66", "#0a1f0a")
            messagebox.showinfo("Saved", f"CSV report saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # ── PDF export ────────────────────────────────────────────────────────────
    def _save_pdf(self):
        if not PDF_OK:
            messagebox.showerror("Error",
                                 "reportlab not installed.\n"
                                 "Run: pip install reportlab")
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            initialfile=f"pcb_report_{ts}.pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            # Save chart as temp image for embedding
            chart_path = path.replace(".pdf", "_chart_tmp.png")
            self.fig.savefig(chart_path, dpi=120, bbox_inches="tight",
                             facecolor=CHART_BG)

            # Save frozen frame as temp image
            snap_path = None
            if self.frozen_frame is not None:
                snap_path = path.replace(".pdf", "_snap_tmp.jpg")
                cv2.imwrite(snap_path, self.frozen_frame)

            doc    = SimpleDocTemplate(path, pagesize=A4,
                                       topMargin=1.5*cm, bottomMargin=1.5*cm,
                                       leftMargin=2*cm, rightMargin=2*cm)
            styles = getSampleStyleSheet()
            story  = []

            # ── Title ──────────────────────────────────────────────────────
            title_style = ParagraphStyle("title",
                                          fontSize=18, leading=22,
                                          alignment=TA_CENTER,
                                          textColor=colors.HexColor("#003366"),
                                          fontName="Helvetica-Bold",
                                          spaceAfter=4)
            sub_style = ParagraphStyle("sub",
                                        fontSize=10, leading=14,
                                        alignment=TA_CENTER,
                                        textColor=colors.HexColor("#555555"),
                                        spaceAfter=12)

            story.append(Paragraph("PCB Defect Inspection Report", title_style))
            story.append(Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
                f"·  Session start: {self.session_start.strftime('%H:%M:%S')}",
                sub_style))
            story.append(HRFlowable(width="100%", thickness=1,
                                     color=colors.HexColor("#ccddee")))
            story.append(Spacer(1, 10))

            # ── Summary table ──────────────────────────────────────────────
            total = self.total_pass + self.total_reject
            pass_pct   = round(self.total_pass   / total * 100) if total else 0
            reject_pct = round(self.total_reject / total * 100) if total else 0

            summary_data = [
                ["Metric", "Value"],
                ["Total Inspected", str(self.total_inspected)],
                ["Passed",          f"{self.total_pass}  ({pass_pct}%)"],
                ["Rejected",        f"{self.total_reject}  ({reject_pct}%)"],
                ["Session Duration",
                 str(datetime.now() - self.session_start).split(".")[0]],
            ]
            t = Table(summary_data, colWidths=[7*cm, 9*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#003366")),
                ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, 0), 11),
                ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f0f4f8"), colors.HexColor("#e0eaf2")]),
                ("FONTSIZE",    (0, 1), (-1, -1), 10),
                ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#ccddee")),
                ("TOPPADDING",  (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]))
            story.append(Paragraph("Inspection Summary",
                                   ParagraphStyle("h2", fontSize=13,
                                                  fontName="Helvetica-Bold",
                                                  textColor=colors.HexColor("#003366"),
                                                  spaceAfter=6)))
            story.append(t)
            story.append(Spacer(1, 14))

            # ── Defect class counts ────────────────────────────────────────
            story.append(Paragraph("Defect Class Breakdown",
                                   ParagraphStyle("h2", fontSize=13,
                                                  fontName="Helvetica-Bold",
                                                  textColor=colors.HexColor("#003366"),
                                                  spaceAfter=6)))
            cls_data = [["Defect Class", "Count"]]
            for cls in ["missing_comp", "missing_tb", "broken_comp", "bent_pin"]:
                cls_data.append([cls.replace("_", " ").title(),
                                  str(self.defect_counts[cls])])
            ct = Table(cls_data, colWidths=[9*cm, 7*cm])
            ct.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#003366")),
                ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, 0), 11),
                ("ALIGN",       (1, 0), (1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f0f4f8"), colors.HexColor("#e0eaf2")]),
                ("FONTSIZE",    (0, 1), (-1, -1), 10),
                ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#ccddee")),
                ("TOPPADDING",  (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]))
            story.append(ct)
            story.append(Spacer(1, 14))

            # ── Chart image ────────────────────────────────────────────────
            story.append(Paragraph("Defect Statistics Chart",
                                   ParagraphStyle("h2", fontSize=13,
                                                  fontName="Helvetica-Bold",
                                                  textColor=colors.HexColor("#003366"),
                                                  spaceAfter=6)))
            story.append(RLImage(chart_path, width=12*cm, height=8*cm))
            story.append(Spacer(1, 14))

            # ── Defect snapshot ────────────────────────────────────────────
            if snap_path:
                story.append(Paragraph("Defect Snapshot (Last Capture)",
                                       ParagraphStyle("h2", fontSize=13,
                                                      fontName="Helvetica-Bold",
                                                      textColor=colors.HexColor("#003366"),
                                                      spaceAfter=6)))
                story.append(RLImage(snap_path, width=14*cm, height=10*cm))
                story.append(Spacer(1, 14))

            # ── Inspection log table ───────────────────────────────────────
            if self.full_log:
                story.append(Paragraph("Inspection Log",
                                       ParagraphStyle("h2", fontSize=13,
                                                      fontName="Helvetica-Bold",
                                                      textColor=colors.HexColor("#003366"),
                                                      spaceAfter=6)))
                log_data = [["#", "Timestamp", "Defect Class",
                              "Confidence", "Status"]]
                for entry in self.full_log[-50:]:   # max 50 rows in PDF
                    log_data.append([
                        str(entry["id"]), entry["timestamp"],
                        entry["class"].replace("_", " ").upper(),
                        str(entry["conf"]), entry["status"]
                    ])
                lt = Table(log_data,
                           colWidths=[1.2*cm, 3.5*cm, 5*cm, 3*cm, 3.5*cm])
                lt.setStyle(TableStyle([
                    ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#003366")),
                    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, 0), 9),
                    ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.HexColor("#f0f4f8"), colors.HexColor("#e0eaf2")]),
                    ("FONTSIZE",    (0, 1), (-1, -1), 8),
                    ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#ccddee")),
                    ("TOPPADDING",  (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ]))
                story.append(lt)

            doc.build(story)

            # Cleanup temp files
            for tmp in [chart_path, snap_path]:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)

            self._set_status(f"✔  PDF saved: {os.path.basename(path)}",
                             "#33cc66", "#0a1f0a")
            messagebox.showinfo("Saved", f"PDF report saved to:\n{path}")

        except Exception as e:
            messagebox.showerror("PDF Error", str(e))

    # ── Log helpers ───────────────────────────────────────────────────────────
    def _append_log(self, detections):
        ts = datetime.now().strftime("%H:%M:%S")
        if not detections:
            self.log_entry_count += 1
            entry = {"id": self.log_entry_count, "timestamp": ts,
                     "class": "—", "conf": "—", "status": "PASS"}
            self.full_log.append(entry)
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
                entry = {"id": self.log_entry_count, "timestamp": ts,
                         "class": cls, "conf": conf, "status": "REJECT"}
                self.full_log.append(entry)
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
        self.full_log.clear()
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
        self._send_serial("START\n")   # auto-start conveyor

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
        self._send_serial("STOP\n")   # auto-stop conveyor

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
                # Auto stop conveyor when defect found
                if self.auto_mode and defect_count > 0:
                    self._send_serial("STOP\n")

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
            for det in detections:
                self.defect_counts[det["class"]] += 1
            now = time.time()
            if now - self._last_beep_time > 2.0:
                self._last_beep_time = now
                play_alert()
            self.after(0, self._redraw_chart)

        self.total_val.configure( text=str(self.total_inspected))
        self.pass_val.configure(  text=str(self.total_pass))
        self.reject_val.configure(text=str(self.total_reject))
        self.fps_val.configure(   text=str(elapsed_ms))

        if self.total_inspected % 30 == 0 or defect_count > 0:
            self.after(0, self._append_log, detections)

    def _set_status(self, text, color="#33cc66", bg="#112200"):
        self.status_frame.configure(fg_color=bg)
        self.status_label.configure(text=text, text_color=color)

    # ── Close ─────────────────────────────────────────────────────────────────
    def on_closing(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        if self.serial_conn:
            self._send_serial("STOP\n")
            self._disconnect_esp32()
        plt.close("all")
        self.destroy()


if __name__ == "__main__":
    app = PCBInspectionHMI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()