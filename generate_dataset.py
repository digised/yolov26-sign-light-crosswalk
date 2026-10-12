import cv2
import numpy as np
import os
import re
import math
import random
from collections import Counter

BACKGROUND_DIR = r"E:\YOLO\SignDetection\yolov8-sign-detection-main\assets\backgrounds"
SIGNS_DIR = r"E:\YOLO\SignDetection\yolov8-sign-detection-main\assets\signs"
DISTRACTOR_DIR = r"E:\YOLO\SignDetection\yolov8-sign-detection-main\assets\distractors"
LAMPS_DIR = r"E:\YOLO\SignDetection\yolov8-sign-detection-main\assets\lights"
DATASET_DIR = r"E:\YOLO\SignDetection\yolov8-sign-detection-main\output"

IMG_TRAIN_DIR = os.path.join(DATASET_DIR, "images/train")
IMG_VAL_DIR = os.path.join(DATASET_DIR, "images/val")
LBL_TRAIN_DIR = os.path.join(DATASET_DIR, "labels/train")
LBL_VAL_DIR = os.path.join(DATASET_DIR, "labels/val")

TOTAL_IMAGES = 6000
OUTPUT_SIZE = (640, 640)
TRAIN_RATIO = 0.85

RNG = random.Random(42)

FINAL_CLASSES = [
    "barred", "crossWalk", "NoPassB", "NoPassE", "park", "priority",
    "stop", "straight", "tunnelB", "tunnelE", "turn", "slope",
    "school", "speed", "CrossWalk", "Red", "Green"
]

CLASS_WEIGHTS = {
    "CrossWalk": 1.2,
    "crossWalk": 1.0,
}

FILENAME_OVERRIDES = {
    "turnL": "turn", "turnR": "turn",
    "uphill": "slope", "downhill": "slope",
    "speedBump": "speed",
    "CrossWalk01": "CrossWalk", "CrossWalk02": "CrossWalk", "CrossWalk03": "CrossWalk",
}

SIGN_DISTRACTOR_PREFIXES = {
    "straight": ["bc"],
    "turn": ["bc"],
    "barred": ["rc"],
    "NoPassB": ["rc"],
    "slope": ["t"],
    "tunnelB": ["t"],
    "school": ["t"],
    "speed": ["t"],
    "priority": ["t"],
    "park": ["bs"],
    "crossWalk": ["bs"],
    "stop": ["s"],
    "tunnelE": ["e"],
    "NoPassE": ["e"],
    "Red": ["light"],
    "Green": ["light"],
}
CONFUSABLE_BOOST = 5.0

def get_class_from_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    if name in FILENAME_OVERRIDES:
        return FILENAME_OVERRIDES[name]
    for cls in FINAL_CLASSES:
        if cls.lower() == name.lower():
            return cls
    return None

def distractor_prefix(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"([a-zA-Z]+)", stem)
    return m.group(1).lower() if m else "misc"

def lamp_weight(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", stem)
    num = int(m.group(1)) if m else 0
    if num == 1:
        return 3.0
    else:
        return 1.0

def safe_read(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def read_fg_alpha(path):
    raw = safe_read(path)
    if raw is None:
        return None, None
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:
        fg = raw[:, :, :3].copy()
        alpha = raw[:, :, 3].copy()
    elif raw.shape[2] == 3:
        fg = raw.copy()
        alpha = np.full(raw.shape[:2], 255, dtype=np.uint8)
    else:
        return None, None
    return fg, alpha

def exposure(img):
    return np.clip(img * RNG.uniform(0.6, 1.4), 0, 255).astype(np.uint8)

def gamma_correct(img):
    g = RNG.uniform(0.6, 1.6)
    table = np.array([(i / 255.0) ** (1 / g) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def salt_pepper_noise(img, amount=0.02, salt_vs_pepper=0.5):
    out = img.copy()
    h, w = img.shape[:2]
    num_pixels = int(amount * h * w)
    salt_count = int(num_pixels * salt_vs_pepper)
    pepper_count = num_pixels - salt_count
    ys = np.random.randint(0, h, salt_count)
    xs = np.random.randint(0, w, salt_count)
    out[ys, xs] = 255
    ys = np.random.randint(0, h, pepper_count)
    xs = np.random.randint(0, w, pepper_count)
    out[ys, xs] = 0
    return out

def speckle_noise(img):
    noise = np.random.normal(0, RNG.uniform(0.05, 0.20), img.shape)
    out = img.astype(np.float32) * (1 + noise)
    return np.clip(out, 0, 255).astype(np.uint8)

def poisson_noise(img):
    scale = RNG.uniform(0.5, 2.0)
    vals = 2 ** np.ceil(np.log2(max(2, len(np.unique(img)))))
    noisy = np.random.poisson(img.astype(np.float32) * scale * vals) / float(vals * scale)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_noise(img):
    if RNG.random() < 0.55:
        kind = RNG.choices(
            ["gaussian", "salt_pepper", "speckle", "poisson", "gaussian_sp"],
            weights=[30, 25, 20, 10, 15]
        )[0]
        if kind == "gaussian":
            sigma = RNG.uniform(4, 18)
            noise = np.random.normal(0, sigma, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif kind == "salt_pepper":
            amount = RNG.uniform(0.005, 0.03)
            img = salt_pepper_noise(img, amount=amount, salt_vs_pepper=RNG.uniform(0.3, 0.7))
        elif kind == "speckle":
            img = speckle_noise(img)
        elif kind == "poisson":
            img = poisson_noise(img)
        elif kind == "gaussian_sp":
            sigma = RNG.uniform(3, 8)
            noise = np.random.normal(0, sigma, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            img = salt_pepper_noise(img, amount=RNG.uniform(0.002, 0.012), salt_vs_pepper=0.5)
    return img

def motion_blur(img):
    if RNG.random() < 0.4:
        k = RNG.choice([3, 5, 7])
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1
        kernel /= k
        img = cv2.filter2D(img, -1, kernel)
    return img

def lens_blur(img):
    if RNG.random() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def fog(img):
    if RNG.random() < 0.2:
        fog_layer = np.full_like(img, 220)
        img = cv2.addWeighted(img, 0.82, fog_layer, 0.18, 0)
    return img

def color_temperature_shift(img):
    if RNG.random() < 0.5:
        shift = RNG.choice(["warm", "cool", "overcast"])
        img = img.astype(np.float32)
        if shift == "warm":
            img[:, :, 2] = np.clip(img[:, :, 2] * RNG.uniform(1.1, 1.3), 0, 255)
            img[:, :, 0] = np.clip(img[:, :, 0] * RNG.uniform(0.8, 0.95), 0, 255)
        elif shift == "cool":
            img[:, :, 0] = np.clip(img[:, :, 0] * RNG.uniform(1.1, 1.3), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * RNG.uniform(0.8, 0.95), 0, 255)
        else:
            img = np.clip(img * RNG.uniform(0.75, 0.9), 0, 255)
        img = img.astype(np.uint8)
    return img

def sun_glare(img):
    if RNG.random() < 0.25:
        H, W = img.shape[:2]
        cx = RNG.choice([0, W])
        cy = RNG.choice([0, H])
        mask = np.zeros((H, W), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (int(W * RNG.uniform(0.3, 0.6)), int(H * RNG.uniform(0.3, 0.6))), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (101, 101), 0)
        strength = RNG.uniform(0.15, 0.40)
        glare = np.full_like(img, 255, dtype=np.float32)
        img = img.astype(np.float32)
        img = np.clip(img * (1 - strength * mask[:, :, None]) + glare * (strength * mask[:, :, None]), 0, 255).astype(np.uint8)
    return img

def shadow_patch(img):
    if RNG.random() < 0.35:
        H, W = img.shape[:2]
        x1 = RNG.randint(0, W // 2)
        y1 = RNG.randint(0, H // 2)
        x2 = RNG.randint(W // 2, W)
        y2 = RNG.randint(H // 2, H)
        shadow = img.astype(np.float32)
        shadow[y1:y2, x1:x2] *= RNG.uniform(0.3, 0.65)
        img = shadow.astype(np.uint8)
    return img

def apply_scene_effects(img):
    img = color_temperature_shift(img)
    img = exposure(img)
    img = gamma_correct(img)
    img = sun_glare(img)
    img = shadow_patch(img)
    img = add_noise(img)
    img = motion_blur(img)
    img = lens_blur(img)
    img = fog(img)
    return img

def apply_sign_effects(fg, alpha):
    fg = fg.astype(np.float32)
    for c in range(3):
        fg[:, :, c] = np.clip(fg[:, :, c] * RNG.uniform(0.85, 1.15), 0, 255)
    noise = np.random.normal(0, RNG.uniform(2, 12), fg.shape)
    mask3 = (alpha[:, :, None] / 255.0)
    fg = np.clip(fg + noise * mask3, 0, 255)
    return fg.astype(np.uint8)

def apply_lamp_effects(fg, alpha):
    fg = fg.astype(np.float32)
    brightness = RNG.uniform(0.9, 1.1)
    fg = np.clip(fg * brightness, 0, 255)
    noise = np.random.normal(0, RNG.uniform(1, 4), fg.shape)
    mask3 = (alpha[:, :, None] / 255.0)
    fg = np.clip(fg + noise * mask3, 0, 255)
    return fg.astype(np.uint8)

def sign_reflection(fg, alpha):
    if RNG.random() > 0.40:
        return fg
    h, w = fg.shape[:2]
    streak = np.zeros((h, w), dtype=np.float32)
    if RNG.random() < 0.5:
        x0 = RNG.randint(0, w)
        y0 = 0
    else:
        x0 = 0
        y0 = RNG.randint(0, h)
    angle_deg = RNG.uniform(-60, 60)
    angle_rad = np.deg2rad(angle_deg)
    length = int(max(h, w) * RNG.uniform(0.4, 0.9))
    x1 = int(x0 + length * np.cos(angle_rad))
    y1 = int(y0 + length * np.sin(angle_rad))
    thickness = RNG.randint(max(1, int(min(h, w) * 0.06)), max(2, int(min(h, w) * 0.22)))
    cv2.line(streak, (x0, y0), (x1, y1), 1.0, thickness)
    blur_k = max(3, int(thickness * 2) | 1)
    streak = cv2.GaussianBlur(streak, (blur_k, blur_k), 0)
    sign_mask = (alpha / 255.0)
    streak *= sign_mask
    strength = RNG.uniform(0.06, 0.18)
    fg_f = fg.astype(np.float32)
    white = np.full_like(fg_f, 255.0)
    fg_f = np.clip(fg_f * (1 - streak[:, :, None] * strength) + white * (streak[:, :, None] * strength), 0, 255)
    return fg_f.astype(np.uint8)

def sign_blur(fg, alpha, distance_factor=0.5):
    distance_factor = min(1.0, max(0.0, distance_factor))
    probability = 0.65 - 0.35 * distance_factor
    if RNG.random() < probability:
        kind = RNG.choice(["motion", "gaussian"])
        if distance_factor < 0.35:
            kernel_choices = [7, 9, 11, 13, 15]
        elif distance_factor < 0.7:
            kernel_choices = [5, 7, 9, 11]
        else:
            kernel_choices = [3, 5, 7]
        k = RNG.choice(kernel_choices)
        if kind == "motion":
            kernel = np.zeros((k, k))
            if RNG.random() < 0.5:
                kernel[k // 2, :] = 1
            else:
                kernel[:, k // 2] = 1
            kernel /= k
            fg = cv2.filter2D(fg, -1, kernel)
            alpha = cv2.filter2D(alpha, -1, kernel)
        else:
            fg = cv2.GaussianBlur(fg, (k, k), 0)
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return fg, alpha

def resize_keep(img, alpha, target_w):
    h, w = img.shape[:2]
    ratio = target_w / w
    new_h = max(1, int(h * ratio))
    return (cv2.resize(img, (target_w, new_h)), cv2.resize(alpha, (target_w, new_h)))

def rotate_with_alpha(img, alpha, angle):
    """
    Rotate while EXPANDING the canvas so no corner of the sign is ever
    clipped by the sprite's own bounding box. Fixes the previous bug
    where rotating a square sprite in a same-size canvas cut off its
    corners (since a rotated rectangle's bounding box is larger than
    the original one).
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(math.ceil((h * sin) + (w * cos)))
    new_h = int(math.ceil((h * cos) + (w * sin)))

    # shift the transform so the rotated content is centered in the new canvas
    M[0, 2] += (new_w / 2) - w / 2
    M[1, 2] += (new_h / 2) - h / 2

    img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    alpha = cv2.warpAffine(alpha, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, alpha

def perspective_transform(img, alpha, shift_range=(0.05, 0.20)):
    h, w = img.shape[:2]
    shift = RNG.uniform(*shift_range)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[shift*w,0],[w-shift*w,shift*h],[w,h-shift*h],[0,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    alpha = cv2.warpPerspective(alpha, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, alpha

def perspective_crosswalk(img, alpha):
    factor = RNG.uniform(1.8, 3.2)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * factor), h))
    alpha = cv2.resize(alpha, (int(w * factor), h))
    h, w = img.shape[:2]
    top_shrink = RNG.uniform(0.25, 0.55)
    margin = int(w * top_shrink / 2)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[margin,0],[w-margin,0],[w,h],[0,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    alpha = cv2.warpPerspective(alpha, M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, alpha

def random_sign_transform(img, alpha, mode="normal"):
    if mode == "crosswalk":
        img, alpha = perspective_crosswalk(img, alpha)
        return rotate_with_alpha(img, alpha, RNG.uniform(-5, 5))
    if mode == "light":
        angle = RNG.uniform(-15, 15)
        img, alpha = rotate_with_alpha(img, alpha, angle)
        if RNG.random() < 0.6:
            img, alpha = perspective_transform(img, alpha, shift_range=(0.03, 0.12))
        return img, alpha
    if mode == "directional":
        angle = RNG.uniform(-10, 10)
        img, alpha = rotate_with_alpha(img, alpha, angle)
        if RNG.random() < 0.4:
            img, alpha = perspective_transform(img, alpha)
        return img, alpha
    angle = RNG.uniform(-50, 50)
    img, alpha = rotate_with_alpha(img, alpha, angle)
    if RNG.random() < 0.85:
        img, alpha = perspective_transform(img, alpha, shift_range=(0.05, 0.25))
    return img, alpha

def fit_within_frame(img, alpha, max_w, max_h):
    """
    Guarantee that a foreground sprite never has to be cropped when it
    is pasted into the scene: if it ended up larger than the space it
    is allowed to occupy (e.g. crosswalk widened by perspective, or a
    sign whose rotated bounding box grew), scale it DOWN proportionally
    (aspect ratio preserved, full shape kept) instead of letting the
    paste step silently cut off part of it.
    """
    h, w = img.shape[:2]
    max_w = max(1, int(max_w))
    max_h = max(1, int(max_h))
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h))
        alpha = cv2.resize(alpha, (new_w, new_h))
    return img, alpha

def paste(bg, fg, alpha, x, y):
    bg_h, bg_w = bg.shape[:2]
    fg_h, fg_w = fg.shape[:2]
    x1, x2 = max(0, x), min(bg_w, x + fg_w)
    y1, y2 = max(0, y), min(bg_h, y + fg_h)
    fg_x1, fg_x2 = max(0, -x), min(fg_w, bg_w - x)
    fg_y1, fg_y2 = max(0, -y), min(fg_h, bg_h - y)
    if x1 >= x2 or y1 >= y2:
        return bg
    roi = bg[y1:y2, x1:x2]
    fg_c = fg[fg_y1:fg_y2, fg_x1:fg_x2]
    a_c = alpha[fg_y1:fg_y2, fg_x1:fg_x2]
    a = a_c[:, :, None] / 255.0
    bg[y1:y2, x1:x2] = (fg_c * a + roi * (1 - a)).astype(np.uint8)
    return bg

def spot_light_on_sign(img, x, y, w, h):
    if RNG.random() < 0.45:
        H, W = img.shape[:2]
        rw = int(w * RNG.uniform(0.10, 0.50))
        rh = int(h * RNG.uniform(0.05, 0.30))
        blur_size = max(15, int(max(rw, rh) * 2) | 1)
        margin = blur_size
        rx1 = max(0, x - margin)
        ry1 = max(0, y - margin)
        rx2 = min(W, x + w + margin)
        ry2 = min(H, y + h + margin)
        roi = img[ry1:ry2, rx1:rx2]
        rH, rW = roi.shape[:2]
        cx = (x - rx1) + RNG.randint(int(w * 0.2), int(w * 0.8))
        cy = (y - ry1) + RNG.randint(int(h * 0.2), int(h * 0.8))
        mask = np.zeros((rH, rW), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (max(1,rw), max(1,rh)), RNG.uniform(0, 180), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        strength = RNG.uniform(0.03, 0.55)
        roi_f = roi.astype(np.float32)
        reflection = np.full_like(roi_f, 255)
        roi_f = (roi_f * (1 - mask[:, :, None] * strength) + reflection * (mask[:, :, None] * strength))
        img[ry1:ry2, rx1:rx2] = np.clip(roi_f, 0, 255).astype(np.uint8)
    return img

def weighted_sample_without_replacement(population, weights, k):
    k = min(k, len(population))
    keyed = []
    for item, w in zip(population, weights):
        w = max(float(w), 1e-9)
        u = RNG.random()
        key = u ** (1.0 / w)
        keyed.append((key, item))
    keyed.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in keyed[:k]]

def generate():
    os.makedirs(IMG_TRAIN_DIR, exist_ok=True)
    os.makedirs(IMG_VAL_DIR, exist_ok=True)
    os.makedirs(LBL_TRAIN_DIR, exist_ok=True)
    os.makedirs(LBL_VAL_DIR, exist_ok=True)

    backgrounds = [
        os.path.join(BACKGROUND_DIR, f)
        for f in os.listdir(BACKGROUND_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    signs_all = [
        os.path.join(SIGNS_DIR, f)
        for f in os.listdir(SIGNS_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    distractors = [
        os.path.join(DISTRACTOR_DIR, f)
        for f in os.listdir(DISTRACTOR_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    lamps_all = []
    if os.path.exists(LAMPS_DIR):
        lamps_all = [
            os.path.join(LAMPS_DIR, f)
            for f in os.listdir(LAMPS_DIR)
            if f.lower().endswith(".png")
        ]

    if not backgrounds:
        raise RuntimeError(f"No backgrounds found in {BACKGROUND_DIR}")
    if not signs_all:
        raise RuntimeError(f"No signs found in {SIGNS_DIR}")

    sign_classes_map = {}
    for p in signs_all:
        c = get_class_from_filename(p)
        if c is None:
            continue
        sign_classes_map.setdefault(c, []).append(p)
    sign_class_names = list(sign_classes_map.keys())
    sign_class_weights = [CLASS_WEIGHTS.get(c, 1.0) for c in sign_class_names]

    distractor_prefixes = [distractor_prefix(p) for p in distractors]
    family_counts = Counter(distractor_prefixes)
    distractor_base_weights = [1.0 / family_counts[pfx] for pfx in distractor_prefixes]

    lamp_weights = [lamp_weight(p) for p in lamps_all]

    n_bg = len(backgrounds)
    reps = math.ceil(TOTAL_IMAGES / n_bg)
    bg_sequence = (backgrounds * reps)[:TOTAL_IMAGES]
    RNG.shuffle(bg_sequence)

    for idx in range(TOTAL_IMAGES):
        if idx % 50 == 0 or idx == TOTAL_IMAGES - 1:
            print(f"[{idx + 1}/{TOTAL_IMAGES}] generating...")

        bg = safe_read(bg_sequence[idx])
        if bg is None:
            continue
        if bg.ndim == 3 and bg.shape[2] == 4:
            bg = bg[:, :, :3]
        bg = cv2.resize(bg, OUTPUT_SIZE)
        H, W = bg.shape[:2]
        composed = bg.copy()
        labels = []
        occupancy_mask = np.zeros((H, W), dtype=np.uint8)
        placed_sign_classes = set()

        # vertical band reserved for crosswalks (near the bottom of the frame)
        crosswalk_y_min = int(H * 0.6)
        crosswalk_y_max = H
        crosswalk_available_h = crosswalk_y_max - crosswalk_y_min

        # vertical band reserved for traffic lights (upper part of the frame)
        light_y_max = int(H * 0.6)

        n_signs = RNG.randint(3, 6)
        chosen_classes = weighted_sample_without_replacement(
            sign_class_names, sign_class_weights, n_signs
        )
        chosen = [RNG.choice(sign_classes_map[c]) for c in chosen_classes]

        for path in chosen:
            cname = get_class_from_filename(path)
            if cname is None:
                continue

            fg, alpha = read_fg_alpha(path)
            if fg is None:
                continue

            is_crosswalk = (cname == "CrossWalk")

            if is_crosswalk:
                fg = fg.astype(np.float32)
                dark_factor = RNG.uniform(0.7, 1.0)
                fg = np.clip(fg * dark_factor, 0, 255).astype(np.uint8)
                crosswalk_clear = RNG.random() < 0.85
                if not crosswalk_clear:
                    fg = apply_sign_effects(fg, alpha)
                    fg = sign_reflection(fg, alpha)
            else:
                fg = apply_sign_effects(fg, alpha)
                fg = sign_reflection(fg, alpha)

            if cname == "CrossWalk":
                target_w_frac = RNG.uniform(0.20, 0.45)
            else:
                target_w_frac = RNG.uniform(0.12, 0.40)
            target_w = int(W * target_w_frac)

            fg, alpha = resize_keep(fg, alpha, target_w)

            mode = "crosswalk" if is_crosswalk else "directional" if cname in ("turn", "straight") else "normal"
            fg, alpha = random_sign_transform(fg, alpha, mode)

            # --- no-crop guarantee -------------------------------------------------
            # After rotation/perspective the sprite's bounding box can grow beyond
            # the space it is allowed to occupy. Scale it back down (keeping its
            # full shape and aspect ratio) instead of letting `paste` truncate it.
            if is_crosswalk:
                fg, alpha = fit_within_frame(fg, alpha, W, crosswalk_available_h)
            else:
                fg, alpha = fit_within_frame(fg, alpha, W, H)
            # -------------------------------------------------------------------

            if not is_crosswalk:
                distance_factor = (target_w_frac - 0.12) / (0.40 - 0.12)
                fg, alpha = sign_blur(fg, alpha, distance_factor)
                fg, alpha = fit_within_frame(fg, alpha, W, H)

            sh, sw = fg.shape[:2]

            for _ in range(30):
                if cname == "CrossWalk":
                    px = RNG.randint(0, max(0, W - sw))
                    py = RNG.randint(crosswalk_y_min, max(crosswalk_y_min, crosswalk_y_max - sh))
                else:
                    px = RNG.randint(0, max(0, W - sw))
                    py = RNG.randint(0, max(0, H - sh))

                if np.sum(occupancy_mask[py:py+sh, px:px+sw]) == 0:
                    occupancy_mask[py:py+sh, px:px+sw] = 1
                    composed = spot_light_on_sign(composed, px, py, sw, sh)
                    composed = paste(composed, fg, alpha, px, py)
                    cx = (px + sw / 2) / W
                    cy = (py + sh / 2) / H
                    labels.append(f"{FINAL_CLASSES.index(cname)} {cx:.6f} {cy:.6f} {sw/W:.6f} {sh/H:.6f}")
                    placed_sign_classes.add(cname)
                    break

        if lamps_all:
            n_lights = RNG.choices([0, 1, 2], weights=[30, 50, 20])[0]
            for _ in range(n_lights):
                l_path = RNG.choices(lamps_all, weights=lamp_weights, k=1)[0]
                filename = os.path.basename(l_path).lower()
                if "green" in filename:
                    cname = "Green"
                elif "red" in filename:
                    cname = "Red"
                else:
                    continue

                fg, alpha = read_fg_alpha(l_path)
                if fg is None:
                    continue

                fg = apply_lamp_effects(fg, alpha)
                target_w = int(W * RNG.uniform(0.07, 0.14))
                fg, alpha = resize_keep(fg, alpha, target_w)
                fg, alpha = random_sign_transform(fg, alpha, "light")

                # no-crop guarantee for lamps (reserved to the upper band of the frame)
                fg, alpha = fit_within_frame(fg, alpha, W, light_y_max)

                sh, sw = fg.shape[:2]

                for _ in range(30):
                    px = RNG.randint(0, max(0, W - sw))
                    py = RNG.randint(0, max(0, light_y_max - sh))

                    if np.sum(occupancy_mask[py:py+sh, px:px+sw]) == 0:
                        occupancy_mask[py:py+sh, px:px+sw] = 1
                        composed = paste(composed, fg, alpha, px, py)
                        cx = (px + sw / 2) / W
                        cy = (py + sh / 2) / H
                        labels.append(f"{FINAL_CLASSES.index(cname)} {cx:.6f} {cy:.6f} {sw/W:.6f} {sh/H:.6f}")
                        placed_sign_classes.add(cname)
                        break

        n_dist = RNG.choices([1, 2, 3, 4], weights=[30, 35, 20, 15])[0]

        if n_dist and distractors:
            relevant_prefixes = set()
            for c in placed_sign_classes:
                relevant_prefixes.update(SIGN_DISTRACTOR_PREFIXES.get(c, []))

            if relevant_prefixes:
                dist_weights = [
                    distractor_base_weights[i] * (CONFUSABLE_BOOST if distractor_prefixes[i] in relevant_prefixes else 1.0)
                    for i in range(len(distractors))
                ]
            else:
                dist_weights = distractor_base_weights

            chosen_distractors = RNG.choices(distractors, weights=dist_weights, k=n_dist)

            for d_path in chosen_distractors:
                d_fg, d_alpha = read_fg_alpha(d_path)
                if d_fg is None:
                    continue

                dist_size = RNG.randint(30, 80)
                d_fg = cv2.resize(d_fg, (dist_size, dist_size))
                d_alpha = cv2.resize(d_alpha, (dist_size, dist_size))

                dpx = RNG.randint(0, max(0, W - dist_size))
                dpy = RNG.randint(0, max(0, H - dist_size))

                if np.sum(occupancy_mask[dpy:dpy+dist_size, dpx:dpx+dist_size]) == 0:
                    composed = paste(composed, d_fg, d_alpha, dpx, dpy)
                    occupancy_mask[dpy:dpy+dist_size, dpx:dpx+dist_size] = 1

        composed = apply_scene_effects(composed)

        path_set = (IMG_TRAIN_DIR, LBL_TRAIN_DIR) if RNG.random() < TRAIN_RATIO else (IMG_VAL_DIR, LBL_VAL_DIR)

        cv2.imwrite(os.path.join(path_set[0], f"{idx}.jpg"), composed, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(path_set[1], f"{idx}.txt"), "w") as f:
            f.write("\n".join(labels))

    print("DONE")

if __name__ == "__main__":
    generate()