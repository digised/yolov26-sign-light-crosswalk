import time, glob, os, platform
import yaml, torch
from ultralytics import YOLO

N_IMAGES = 200
ROOT = os.path.dirname(os.path.abspath(__file__))


def find_best_pt():
    cands = glob.glob(os.path.join(ROOT, "**", "best.pt"), recursive=True)
    if not cands:
        raise SystemExit("best.pt not found. Set the weights path manually.")
    main_run = [c for c in cands if "150ep" in c]
    pool = main_run or cands
    return max(pool, key=os.path.getmtime)


def find_val_images():
    imgs = []
    for yml in glob.glob(os.path.join(ROOT, "**", "dataset.yaml"), recursive=True):
        try:
            cfg = yaml.safe_load(open(yml, encoding="utf-8"))
            base = cfg.get("path", os.path.dirname(yml))
            if not os.path.isabs(base):
                base = os.path.join(os.path.dirname(yml), base)
            val = cfg["val"]
            val = val if os.path.isabs(val) else os.path.join(base, val)
            if os.path.isdir(val):
                for ext in ("jpg", "jpeg", "png"):
                    imgs += glob.glob(os.path.join(val, "**", f"*.{ext}"), recursive=True)
            elif os.path.isfile(val):
                imgs += [l.strip() for l in open(val, encoding="utf-8") if l.strip()]
            if imgs:
                print("dataset.yaml:", yml)
                break
        except Exception as e:
            print("skip", yml, e)
    if not imgs:
        for d in glob.glob(os.path.join(ROOT, "**", "val*"), recursive=True):
            if os.path.isdir(d) and "runs" not in d:
                for ext in ("jpg", "jpeg", "png"):
                    imgs += glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True)
                if imgs:
                    break
    if not imgs:
        raise SystemExit("No validation images found. Set the image folder manually.")
    return sorted(imgs)[:N_IMAGES]


def bench(model, imgs, device, half, n_warm=20):
    for p in imgs[:n_warm]:
        model.predict(p, device=device, half=half, imgsz=640, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in imgs:
        model.predict(p, device=device, half=half, imgsz=640, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / len(imgs)
    return dt * 1000, 1 / dt


def main():
    print("=" * 50)
    print("CPU:", platform.processor())
    print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    print("CPU threads:", torch.get_num_threads())
    print("=" * 50)

    W = find_best_pt()
    imgs = find_val_images()
    print("Weights:", W)
    print("Images used:", len(imgs))
    model = YOLO(W)

    for name, dev, half in [("GPU FP16", 0, True),
                            ("GPU FP32", 0, False),
                            ("CPU FP32", "cpu", False)]:
        ms, fps = bench(model, imgs, dev, half)
        print(f"{name}: {ms:.1f} ms/img | {fps:.1f} FPS")


if __name__ == "__main__":
    main()