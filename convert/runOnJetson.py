import cv2
from ultralytics import YOLO

model = YOLO("best.engine", task="detect")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(source=frame, conf=0.5, imgsz=640)
    annotated_frame = results[0].plot()

    cv2.imshow("Jetson Nano YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()