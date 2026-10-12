import cv2
cv2.setNumThreads(0)

from ultralytics import YOLO
import torch


def main():

    print("=" * 60)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

    if not torch.cuda.is_available():
        print("\nERROR: CUDA/GPU is NOT available!")
        print("Training stopped. The model will NOT train on CPU.")
        return

    device = 0

    print("GPU:", torch.cuda.get_device_name(0))
    print("Using device:", device)
    print("GPU Memory:", round(
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
    ), "GB")
    print("=" * 60)

    DATASET_PATH = (
        r"E:\YOLO\SignDetection\yolov8-sign-detection-main\dataset.yaml"
    )

    PROJECT_PATH = (
        r"E:\YOLO\SignDetection\yolov8-sign-detection-main\runs\detect"
    )

    MODEL_YAML = "yolo26n.yaml"

    model = YOLO(MODEL_YAML)

    results = model.train(
        data=DATASET_PATH,
        project=PROJECT_PATH,
        name="YOLO26_Scratch_150ep",

        epochs=150,
        patience=40,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        amp=True,
        cache="disk",

        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        freeze=None,

        mosaic=0.7,
        close_mosaic=10,
        mixup=0.0,
        degrees=3.0,
        shear=0.0,
        perspective=0.0,
        translate=0.05,
        scale=0.30,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        save=True,
        val=True,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("Training finished successfully!")
    print("=" * 60)

    print("\nResults saved in:")
    print(PROJECT_PATH)


if __name__ == "__main__":
    main()