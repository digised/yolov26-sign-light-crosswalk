# Traffic Perception for Autonomous Car

<img width="1920" height="1920" alt="8" src="https://github.com/user-attachments/assets/3e04e999-0e91-43ff-98a7-0e4623272eb9" />

YOLO-based detector for traffic signs, traffic lights, and crosswalks, built for Team Auren's self-driving car at RoboCup IranOpen 2026 (Self-Driving Car League).

## Repository Contents

| File | Purpose |
|---|---|
| `generate_dataset.py` | Generates a synthetic YOLO-format dataset by compositing sign, light, and distractor sprites onto background images |
| `annotate.py` | Tkinter tool for manual bounding-box annotation, reads and writes YOLO labels |
| `bench_inference.py` | Measures inference speed on GPU and CPU |

## Classes (17)

`barred, crossWalk, NoPassB, NoPassE, park, priority, stop, straight, tunnelB, tunnelE, turn, slope, school, speed, CrossWalk, Red, Green`

`crossWalk` is the crosswalk sign. `CrossWalk` is the road marking, which the generator stretches with perspective and places in the lower part of the frame.

## Dataset

The dataset is synthetic, created by `generate_dataset.py`.

| Item | Value |
|---|---|
| Images | 6,000 at 640x640 |
| Split | random 85% train / 15% val (about 5,100 / 900 images) |
| Labeled instances | about 22,000 in the training set |
| Instances per class | about 1,200 to 1,900 |

How each image is made:

- 3 to 6 signs per image, plus 0 to 2 traffic lights (red or green) in the upper part of the frame
- 1 to 4 distractor sprites per image, with look-alike distractors sampled 5x more often for the placed sign classes
- Sprite transforms: rotation, perspective, brightness, noise, reflection streaks, and distance-dependent blur
- Scene effects: color temperature, exposure, gamma, sun glare, shadow patches, Gaussian / salt-and-pepper / speckle / Poisson noise, motion and lens blur, fog
- YOLO labels are written automatically from the placement of each sprite

`dataset.yaml`:

```yaml
path: <path to generated output folder>
train: images/train
val: images/val

nc: 17
names: ["barred", "crossWalk", "NoPassB", "NoPassE", "park", "priority", "stop", "straight", "tunnelB", "tunnelE", "turn", "slope", "school", "speed", "CrossWalk", "Red", "Green"]
```

## Annotation Tool

`annotate.py` is a desktop tool for labeling images by hand.

- Draw, move, and resize boxes, with a class picker (keyboard shortcuts per class)
- Zoom and pan, undo, delete, and mark an image as having no objects
- Saves YOLO-format `.txt` files into a `labels/` folder next to the images
- Remembers the last image position per folder

```
python annotate.py --folder path/to/images
```

## Training Setup

Values taken from the run's `args.yaml`.

| Setting | Value |
|---|---|
| Model | yolo26n, built from YAML (from scratch) |
| Epochs | 150 (early stopping patience 40) |
| Image size / batch | 640 / 16 |
| Optimizer | auto, cosine LR (lr0 0.01, lrf 0.01) |
| Mixed precision | on (AMP) |
| Augmentation | mosaic 0.7 (off for last 10 epochs), rotation 3 deg, translate 0.05, scale 0.3, HSV; no flips |
| Seed | 0 (deterministic) |
| Framework | PyTorch 2.11, CUDA 12.8, Ultralytics |
| Hardware | RTX 3070 Laptop GPU (8 GB), Intel Core i7-12700H |

## Results

Final epoch (150) on the validation split.

| Metric | Value |
|---|---|
| Precision | 0.998 |
| Recall | 0.997 |
| mAP50 | 0.995 |
| mAP50-95 | 0.982 |
| Total training time | 2.4 hours (about 58 s per epoch, including validation) |

## Inference Speed

Measured on 200 validation images, image size 640, batch size 1. Times include pre-processing and post-processing.

| Device | Precision | ms per image | FPS |
|---|---|---|---|
| RTX 3070 Laptop (8 GB) | FP32 | 13.9 | 72.1 |
| RTX 3070 Laptop (8 GB) | FP16 | 14.4 | 69.3 |
| Intel Core i7-12700H (14 threads) | FP32 | 47.4 | 21.1 |

GPU inference is about 3.4x faster than CPU. FP16 gave no speedup in this setup.

## Usage

Install dependencies:

```
pip install ultralytics opencv-python numpy pillow pyyaml
```

Generate the dataset (set the folder paths at the top of the script first):

```
python generate_dataset.py
```

Train (use the training script in this repo with the settings above).

Benchmark inference (finds `best.pt` and validation images automatically):

```
python bench_inference.py
```

## Limitations

- Training and validation images both come from the same generator, and they share background images and sprites. The metrics therefore measure performance on synthetic data, and the near-perfect mAP reflects that.
- No evaluation on real track footage has been done yet, so real-world accuracy may be lower.
- Speed numbers were measured on a laptop and depend on the hardware.
