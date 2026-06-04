from ultralytics import YOLO
import cv2
import math

# --- CONFIGURATION ---
model_path = 'runs/detect/train_v2/weights/best.pt' 

# Camera ID 
camera_id = 0 
# ---------------------

model = YOLO(model_path)
cap = cv2.VideoCapture(camera_id)

while True:
    success, img = cap.read()
    if not success:
        break

    # Run Inference
    results = model(img, stream=True)

    defect_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Class Name
            cls = int(box.cls[0])
            class_name = model.names[cls]
            conf = math.ceil((box.conf[0] * 100)) / 100

            if class_name == 'short':
                color = (0, 0, 255)       # Red for Short Circuit
                label_text = "CRITICAL: SHORT"
            elif class_name == 'missing_hole':
                color = (0, 165, 255)     # Orange for Missing parts
                label_text = "MISSING PART"
            elif class_name == 'open':
                color = (0, 255, 255)     # Yellow for Open Circuit
                label_text = "OPEN CIRCUIT"
            else:
                color = (0, 0, 255)       # Default Red for any other defect
                label_text = class_name.upper()

            defect_count += 1

            # Draw Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw Label Background & Text
            t_size = cv2.getTextSize(f'{label_text} {conf}', 0, fontScale=0.6, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, color, -1) 
            cv2.putText(img, f'{label_text} {conf}', (x1, y1 - 2), 0, 0.6, (255, 255, 255), 1)

    # DASHBOARD STATS
    if defect_count > 0:
        status_color = (0, 0, 255) # Red banner
        status_text = f"WARNING: {defect_count} DEFECTS FOUND"
    else:
        status_color = (0, 255, 0) # Green banner
        status_text = "STATUS: PCB OK"

    # Draw the banner
    cv2.rectangle(img, (0, 0), (1280, 50), status_color, -1)
    cv2.putText(img, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('PCB Inspection System', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()