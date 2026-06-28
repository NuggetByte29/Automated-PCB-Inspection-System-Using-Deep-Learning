# Automated PCB Inspection System

An AI-powered automated inspection system for detecting surface-mount technology (SMT) defects on PCBs using YOLOv11s and a conveyor belt setup, built as a Final Year Project at **Universiti Teknologi Malaysia (UTM)**.

---

## System Overview

The system uses a USB webcam mounted above a DC motor conveyor belt. When a PCB is detected by an IR proximity sensor, the conveyor stops and the AI model inspects the board in real time. Results are displayed on a custom HMI built with CustomTkinter.
![System Setup](assets/system_setup.jpg)

---

## AI Model

| Property | Details |
|---|---|
| Model | YOLOv11s |
| Classes | `bent_pin`, `broken_comp`, `missing_comp`, `missing_tb` |
| mAP@0.5 | 96.3% |
| Precision | 95.9% |
| Recall | 91.4% |
| Inference Speed | 11.6ms |
| Training Epochs | 60 |
| Dataset Size | 160 custom-annotated images |

---

## Hardware Stack

| Component | Details |
|---|---|
| Microcontroller | Arduino Uno |
| Camera | USB Webcam (connected to laptop) |
| Conveyor | DC motor + motor driver |
| Trigger Sensor | IR proximity sensor (pin 2) |
| Motor Control Pins | IN1 = pin 4, IN2 = pin 5 |
| Baud Rate | 115200 |

---

## Software Stack

| Tool | Purpose |
|---|---|
| Python | Main application |
| Ultralytics YOLOv11s | Defect detection |
| CustomTkinter | HMI / GUI |
| pyserial | Arduino communication |
| matplotlib | Embedded bar chart |
| reportlab | PDF report export |
| Label Studio | Dataset annotation |

---

## Project Structure

```
fyp_project/
│
├── pcb_hmi_autofreeze1.py    # Main HMI application
├── run_pcb_dashboard.py      # Launcher script
├── train_val_split.py        # Dataset split utility
├── data.yaml                 # YOLO dataset config
│
├── data/                     # Dataset images (excluded from repo)
├── my_dataset/               # Raw dataset (excluded from repo)
├── runs/                     # YOLO training outputs (excluded from repo)
└── *.pt                      # Model weights (excluded from repo)
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/NuggetByte29/Automated-PCB-Inspection-System-Using-Deep-Learning.git
cd Automated-PCB-Inspection-System-Using-Deep-Learning
```

### 2. Install dependencies
```bash
pip install ultralytics customtkinter pyserial matplotlib reportlab
```

### 3. Download model weights
Download `yolo11s.pt` separately and place it in the project root.

> Model weights are excluded from this repo due to file size.

### 4. Connect Arduino
- Flash `pcb_conveyor_uno.ino` to your Arduino Uno
- Connect via USB and note the COM port (e.g. `COM3`)

### 5. Run the HMI
```bash
python run_pcb_dashboard.py
```

---

## Experimental Results (30-PCB Validation)

| Metric | Result |
|---|---|
| PASS/REJECT Accuracy | 100% |
| Defect-Level Recall | 93.9% |
| Exact Defect Count Accuracy | 83.3% |
| Mean Confidence Score | 0.798 |
| Ground Truth Defects | 131 |
| Detected Defects | 123 |

### Per-Class Recall
| Class | Recall |
|---|---|
| `missing_comp` | 91.7% |
| `missing_tb` | 100% |
| `broken_comp` | 100% |
| `bent_pin` | 92.3% |

---

## 🔁 Serial Communication Protocol

| Sender | Signal | Meaning |
|---|---|---|
| Python → Arduino | `START` | Begin conveyor |
| Python → Arduino | `STOP` | Stop conveyor |
| Python → Arduino | `SENSOR_BYPASS` | Bypass sensor for 3s |
| Arduino → Python | `PCB_DETECTED` | IR sensor triggered |
| Arduino → Python | `CONVEYOR_ON` | Conveyor moving |
| Arduino → Python | `CONVEYOR_OFF` | Conveyor stopped |
| Arduino → Python | `READY` | System ready |

---

## Author

**ADIB ADLI ALEK BIN CHE ALEK**
B.Eng Electrical-Mechatronics Engineering
Universiti Teknologi Malaysia (UTM)

---

## License

This project is for academic purposes. All rights reserved.
