# ==========================================================
# IMPORTS
# ==========================================================
import cv2
import numpy as np
import depthai as dai  # (unused in webcam mode, but kept for compatibility)
from pathlib import Path
from skimage.feature import hog
import joblib


# ==========================================================
# PATHS (Relative to script location)
# ==========================================================
SCRIPT_DIR = Path(__file__).parent
YOLO_MODEL_PATH = SCRIPT_DIR / "models" / "best.pt"
SVM_MODEL_PATH = SCRIPT_DIR / "models" / "hog_svm_turn_slope.pkl"


# ==========================================================
# CAMERA AUTO-DETECT (Windows friendly)
# ==========================================================
def list_dshow_devices():
    """
    List DirectShow camera device names (Windows).
    Optional dependency: pip install pygrabber
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception:
        return []


def probe_camera_index(max_index=10, backend=cv2.CAP_DSHOW):
    """
    Return list of working camera indices by trying to open and read a frame.
    """
    working = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        ok, _ = cap.read()
        cap.release()
        if ok:
            working.append(i)
    return working


def pick_camera_index(prefer_keywords=("Iriun",), max_index=10):
    """
    Pick best camera index:
    1) Print DirectShow devices if available
    2) Probe indices (CAP_DSHOW)
    3) Prefer an index that likely corresponds to Iriun if possible
    """
    devices = list_dshow_devices()
    if devices:
        print("\n[DirectShow Devices]")
        for i, d in enumerate(devices):
            print(f"  {i}: {d}")
    else:
        print("\n[DirectShow Devices] (not available) Install 'pygrabber' to see device names.")

    working = probe_camera_index(max_index=max_index, backend=cv2.CAP_DSHOW)
    print("\n[Camera Probe] Working indices (CAP_DSHOW):", working if working else "None")

    if not working:
        return None

    if devices:
        preferred_positions = []
        for pos, name in enumerate(devices):
            if any(k.lower() in name.lower() for k in prefer_keywords):
                preferred_positions.append(pos)

        if preferred_positions:
            best = min(working, key=lambda idx: min(abs(idx - p) for p in preferred_positions))
            return best

    return working[0]


# ==========================================================
# LOAD SVM
# ==========================================================
def load_svm_model():
    """Load SVM model and label encoder"""
    if not SVM_MODEL_PATH.exists():
        raise FileNotFoundError(f"SVM model not found at {SVM_MODEL_PATH}")
    return joblib.load(SVM_MODEL_PATH)


# ==========================================================
# HOG FEATURE EXTRACTOR
# ==========================================================
def extract_hog(cv2_img):
    """Extract HOG features from image"""
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features


# ==========================================================
# CLASSICAL SLOPE FALLBACK
# ==========================================================
def classify_hill_classical(cv2_crop):
    """Classify slope using classical image processing as fallback"""
    gray = cv2.cvtColor(cv2_crop, cv2.COLOR_BGR2GRAY)
    arr = 255 - gray

    h, w = arr.shape
    crop = arr[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    crop = cv2.resize(crop, (100, 100))

    left = crop[:, :50]
    right = crop[:, 50:]

    left_dark = np.sum(left)
    right_dark = np.sum(right)

    if right_dark > left_dark * 1.15:
        return "uphill"
    elif left_dark > right_dark * 1.15:
        return "downhill"
    else:
        return "unknown"


# ==========================================================
# FINAL CLASSIFIER (HOG + SVM)
# ==========================================================
def classify_turn_or_slope(crop, sign_type, svm, label_encoder):
    """Classify turn direction or slope using HOG+SVM with fallback"""
    feat = extract_hog(crop)

    pred_idx = svm.predict([feat])[0]
    conf = np.max(svm.predict_proba([feat])[0])

    pred = label_encoder.inverse_transform([pred_idx])[0]

    if sign_type == "turn":
        if pred in ["turnL", "turnR"] and conf > 0.6:
            return pred
        return "unknown"

    if sign_type == "slope":
        if pred in ["uphill", "downhill"] and conf > 0.6:
            return pred

        return classify_hill_classical(crop)

    return "unknown"


# ==========================================================
# YOLOv8 + Webcam
# ==========================================================
class OAKCameraYOLOv8Custom:
    def __init__(self, model_path, conf_thresh=0.5):
        self.model_path = model_path
        self.conf_thresh = conf_thresh

        # لیست کلاس‌ها دقیقاً مطابق با dataset آپدیت شد
        self.classes = [
            "barred", "crossWalk", "NoPassB", "NoPassE", "park", "priority",
            "stop", "straight", "tunnelB", "tunnelE", "turn", "slope",
            "school", "speed", "CrossWalk", "Red", "Green"
        ]

        rng = np.random.default_rng(42)
        self.colors = rng.integers(0, 255, (len(self.classes), 3)).tolist()

        self.svm = None
        self.label_encoder = None

        # NEW (safe): will be overwritten by YOLO model names
        self.model_names = None

    def load_models(self):
        """Load YOLO and SVM models"""
        from ultralytics import YOLO

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")

        self.model = YOLO(str(self.model_path))

        # NEW: sync class list with YOLO model
        self.model_names = self.model.names
        if isinstance(self.model_names, dict):
            self.classes = [self.model_names[i] for i in range(len(self.model_names))]
        else:
            self.classes = list(self.model_names)

        # NEW: regenerate colors based on actual number of classes
        rng = np.random.default_rng(42)
        self.colors = rng.integers(0, 255, (len(self.classes), 3)).tolist()

        self.svm, self.label_encoder = load_svm_model()

        print("✓ YOLOv8 model loaded")
        print("✓ SVM model loaded")
        print(f"✓ Classes detected from model: {len(self.classes)}")

    def process_frame(self, frame):
        """Process frame and return detections"""
        detections = []
        results = self.model(frame, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_thresh:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])

                # FIX: avoid list index out of range
                if cls_id < 0 or cls_id >= len(self.classes):
                    cls_name = str(cls_id)
                else:
                    cls_name = self.classes[cls_id]

                crop = frame[y1:y2, x1:x2]

                det = {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf
                }

                if cls_name in ["turn", "slope"] and crop.size > 0:
                    det["direction"] = classify_turn_or_slope(
                        crop, cls_name, self.svm, self.label_encoder
                    )

                detections.append(det)

        return detections

    def draw(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]

            # FIX: safe color indexing
            if d["class_id"] < len(self.colors):
                color = tuple(map(int, self.colors[d["class_id"]]))
            else:
                color = (0, 255, 0)

            label = d["class_name"]
            if "direction" in d:
                label += f":{d['direction']}"
            label += f" {d['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def run(self):
        """Run using webcam (auto-detect, prefer Iriun)"""
        self.load_models()

        cam_index = pick_camera_index(prefer_keywords=("Iriun",), max_index=10)
        if cam_index is None:
            raise RuntimeError("No available camera devices found by OpenCV (CAP_DSHOW).")

        print(f"\n✅ Using camera index: {cam_index}")
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n🎥 Webcam running...")
        print("Press 'q' to quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera.")
                break

            detections = self.process_frame(frame)
            frame = self.draw(frame, detections)

            cv2.imshow("Traffic Sign Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ==========================================================
# MAIN
# ==========================================================
def main():
    """Main entry point"""
    print("=" * 60)
    print("Traffic Sign Detection System")
    print("Webcam (Auto Detect) + YOLOv8 + HOG + SVM")
    print("=" * 60)

    try:
        oak = OAKCameraYOLOv8Custom(YOLO_MODEL_PATH)
        oak.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure your models are in the 'models' directory:")
        print(f"  - {YOLO_MODEL_PATH}")
        print(f"  - {SVM_MODEL_PATH}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()