import cv2
import numpy as np
import os
import random
from tqdm import tqdm

# ================= PATHS =================
BACKGROUND_DIR  = r"D:\Auren\hooman test\assets\backgrounds"
SIGNS_DIR       = r"D:\Auren\hooman test\assets\signs"
DISTRACTOR_DIR  = r"D:\Auren\everything\tLightCrossWalk\distractors"
DATASET_DIR     = r"D:\Auren\everything\tLightCrossWalk\datasetsign"

IMG_TRAIN_DIR = os.path.join(DATASET_DIR, "images/train")
IMG_VAL_DIR   = os.path.join(DATASET_DIR, "images/val")
LBL_TRAIN_DIR = os.path.join(DATASET_DIR, "labels/train")
LBL_VAL_DIR   = os.path.join(DATASET_DIR, "labels/val")

TOTAL_IMAGES = 2000
OUTPUT_SIZE  = (640, 640)
TRAIN_RATIO  = 0.85

RNG = random.Random(42)

# ================= CLASSES =================
FINAL_CLASSES = [
    "barred", "crossWalk", "NoPassB", "NoPassE", "park", "priority",
    "stop", "straight", "tunnelB", "tunnelE", "turn", "slope",
    "school", "speed", "CrossWalk", "Green", "Red", "Yellow"
]

CLASS_WEIGHTS = {
    "CrossWalk": 1.5,
    "crossWalk": 1.0,
}

def get_class_from_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    mapping = {
        "turnL": "turn", "turnR": "turn",
        "uphill": "slope", "downhill": "slope",
        "cross": "CrossWalk", "cross2": "CrossWalk",
        "crossWalk": "crossWalk"
    }
    if name in mapping:
        return mapping[name]
    for cls in FINAL_CLASSES:
        if cls.lower() == name.lower():
            return cls
    return None

def safe_read(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

# ================= LIGHTING / EFFECTS =================

def exposure(img):
    return np.clip(img * RNG.uniform(0.6, 1.4), 0, 255).astype(np.uint8)

def gamma_correct(img):
    g = RNG.uniform(0.6, 1.6)
    table = np.array([(i / 255.0) ** (1 / g) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def add_noise(img):
    if RNG.random() < 0.2:
        noise = np.random.normal(0, RNG.uniform(3, 10), img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
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
        cv2.ellipse(mask, (cx, cy),
                    (int(W * RNG.uniform(0.3, 0.6)), int(H * RNG.uniform(0.3, 0.6))),
                    0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (101, 101), 0)
        strength = RNG.uniform(0.15, 0.40)
        glare = np.full_like(img, 255, dtype=np.float32)
        img = img.astype(np.float32)
        img = np.clip(
            img * (1 - strength * mask[:, :, None]) +
            glare * (strength * mask[:, :, None]),
            0, 255
        ).astype(np.uint8)
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

# ================= NEW: SIGN LIGHT REFLECTION =================

def sign_reflection(fg, alpha):
    """
    Adds a subtle specular highlight streak across the sign surface,
    simulating light reflecting off a glossy or metallic sign face.
    Only applied inside the sign's opaque region (alpha > 0).
    """
    if RNG.random() > 0.40:
        return fg

    h, w = fg.shape[:2]

    # Build a diagonal streak mask
    streak = np.zeros((h, w), dtype=np.float32)

    # Streak start: random point along the top or left edge
    if RNG.random() < 0.5:
        x0 = RNG.randint(0, w)
        y0 = 0
    else:
        x0 = 0
        y0 = RNG.randint(0, h)

    angle_deg = RNG.uniform(-60, 60)
    angle_rad = np.deg2rad(angle_deg)

    # Draw a thick line as the streak skeleton
    length = int(max(h, w) * RNG.uniform(0.4, 0.9))
    x1 = int(x0 + length * np.cos(angle_rad))
    y1 = int(y0 + length * np.sin(angle_rad))
    thickness = RNG.randint(max(1, int(min(h, w) * 0.06)),
                             max(2, int(min(h, w) * 0.22)))

    cv2.line(streak, (x0, y0), (x1, y1), 1.0, thickness)

    # Soften the edges
    blur_k = max(3, int(thickness * 2) | 1)
    streak = cv2.GaussianBlur(streak, (blur_k, blur_k), 0)

    # Mask to sign's opaque pixels only
    sign_mask = (alpha / 255.0)
    streak *= sign_mask

    strength = RNG.uniform(0.06, 0.18)   # deliberately subtle

    fg_f = fg.astype(np.float32)
    white = np.full_like(fg_f, 255.0)
    fg_f = np.clip(
        fg_f * (1 - streak[:, :, None] * strength) +
        white * (streak[:, :, None] * strength),
        0, 255
    )

    return fg_f.astype(np.uint8)

# ================= TRANSFORMS =================

def resize_keep(img, alpha, target_w):
    h, w = img.shape[:2]
    ratio = target_w / w
    new_h = int(h * ratio)
    return (cv2.resize(img, (target_w, new_h)),
            cv2.resize(alpha, (target_w, new_h)))

def rotate_with_alpha(img, alpha, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img   = cv2.warpAffine(img,   M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    alpha = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, alpha

def perspective_transform(img, alpha):
    h, w = img.shape[:2]
    shift = RNG.uniform(0.05, 0.20)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[shift*w,0],[w-shift*w,shift*h],[w,h-shift*h],[0,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    img   = cv2.warpPerspective(img,   M, (w,h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    alpha = cv2.warpPerspective(alpha, M, (w,h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, alpha

def perspective_crosswalk(img, alpha):
    factor = RNG.uniform(1.8, 3.2)
    h, w = img.shape[:2]
    img   = cv2.resize(img,   (int(w * factor), h))
    alpha = cv2.resize(alpha, (int(w * factor), h))
    h, w = img.shape[:2]
    top_shrink = RNG.uniform(0.25, 0.55)
    margin = int(w * top_shrink / 2)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[margin,0],[w-margin,0],[w,h],[0,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    img   = cv2.warpPerspective(img,   M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    alpha = cv2.warpPerspective(alpha, M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, alpha

def random_sign_transform(img, alpha, mode="normal"):
    if mode == "crosswalk":
        img, alpha = perspective_crosswalk(img, alpha)
        return rotate_with_alpha(img, alpha, RNG.uniform(-10, 10))
    angle = RNG.uniform(-45, 45)
    img, alpha = rotate_with_alpha(img, alpha, angle)
    if RNG.random() < 0.8:
        img, alpha = perspective_transform(img, alpha)
    return img, alpha

# ================= HELPERS =================

def paste(bg, fg, alpha, x, y):
    bg_h, bg_w = bg.shape[:2]
    fg_h, fg_w = fg.shape[:2]
    x1, x2 = max(0, x), min(bg_w, x + fg_w)
    y1, y2 = max(0, y), min(bg_h, y + fg_h)
    fg_x1, fg_x2 = max(0, -x), min(fg_w, bg_w - x)
    fg_y1, fg_y2 = max(0, -y), min(fg_h, bg_h - y)
    if x1 >= x2 or y1 >= y2:
        return bg
    roi   = bg[y1:y2, x1:x2]
    fg_c  = fg[fg_y1:fg_y2, fg_x1:fg_x2]
    a_c   = alpha[fg_y1:fg_y2, fg_x1:fg_x2]
    a     = a_c[:, :, None] / 255.0
    bg[y1:y2, x1:x2] = (fg_c * a + roi * (1 - a)).astype(np.uint8)
    return bg

def spot_light_on_sign(img, x, y, w, h):
    if RNG.random() < 0.45:
        H, W = img.shape[:2]
        cx = x + RNG.randint(int(w * 0.2), int(w * 0.8))
        cy = y + RNG.randint(int(h * 0.2), int(h * 0.8))
        rw = int(w * RNG.uniform(0.15, 0.35))
        rh = int(h * RNG.uniform(0.08, 0.22))
        mask = np.zeros((H, W), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (max(1,rw), max(1,rh)),
                    RNG.uniform(0, 180), 0, 360, 1.0, -1)
        blur_size = max(15, int(max(rw, rh) * 2) | 1)
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        strength = RNG.uniform(0.08, 0.22)
        img_f = img.astype(np.float32)
        reflection = np.full_like(img_f, 255)
        img_f = (img_f * (1 - mask[:, :, None] * strength)
                 + reflection * (mask[:, :, None] * strength))
        img = np.clip(img_f, 0, 255).astype(np.uint8)
    return img

# ================= MAIN =================

def generate():
    os.makedirs(IMG_TRAIN_DIR, exist_ok=True)
    os.makedirs(IMG_VAL_DIR,   exist_ok=True)
    os.makedirs(LBL_TRAIN_DIR, exist_ok=True)
    os.makedirs(LBL_VAL_DIR,   exist_ok=True)

    backgrounds = [
        os.path.join(BACKGROUND_DIR, f)
        for f in os.listdir(BACKGROUND_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ]
    signs_all = [
        os.path.join(SIGNS_DIR, f)
        for f in os.listdir(SIGNS_DIR)
        if f.lower().endswith(".png")
    ]
    distractors = [
        os.path.join(DISTRACTOR_DIR, f)
        for f in os.listdir(DISTRACTOR_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ]

    sign_weights = [
        CLASS_WEIGHTS.get(get_class_from_filename(p), 1.0)
        for p in signs_all
    ]

    for idx in tqdm(range(TOTAL_IMAGES)):
        bg = safe_read(RNG.choice(backgrounds))
        if bg is None:
            continue
        if bg.ndim == 3 and bg.shape[2] == 4:
            bg = bg[:, :, :3]
        bg = cv2.resize(bg, OUTPUT_SIZE)
        H, W = bg.shape[:2]
        composed = bg.copy()
        labels = []
        occupancy_mask = np.zeros((H, W), dtype=np.uint8)

        n_signs = RNG.randint(3, 6)
        chosen  = RNG.choices(signs_all, weights=sign_weights, k=n_signs)

        for path in chosen:
            cname = get_class_from_filename(path)
            rgba  = safe_read(path)
            if rgba is None or rgba.shape[2] < 4:
                continue

            fg    = rgba[:, :, :3].copy()
            alpha = rgba[:, :, 3].copy()

            fg = apply_sign_effects(fg, alpha)
            fg = sign_reflection(fg, alpha)          # <-- subtle highlight streak

            target_w = int(W * RNG.uniform(0.20, 0.45)) if cname == "CrossWalk" \
                       else int(W * RNG.uniform(0.12, 0.40))

            fg, alpha = resize_keep(fg, alpha, target_w)

            mode      = "crosswalk" if cname == "CrossWalk" else "normal"
            fg, alpha = random_sign_transform(fg, alpha, mode)

            sh, sw = fg.shape[:2]
            success = False

            for _ in range(30):
                if cname == "CrossWalk":
                    px = RNG.randint(0, max(0, W - sw))
                    py = RNG.randint(int(H * 0.6), max(int(H * 0.6) + 1, H - sh))
                else:
                    px = RNG.randint(0, max(0, W - sw))
                    py = RNG.randint(0, max(0, H - sh))

                if np.sum(occupancy_mask[py:py+sh, px:px+sw]) == 0:
                    occupancy_mask[py:py+sh, px:px+sw] = 1
                    composed = spot_light_on_sign(composed, px, py, sw, sh)
                    composed = paste(composed, fg, alpha, px, py)
                    cx = (px + sw / 2) / W
                    cy = (py + sh / 2) / H
                    labels.append(
                        f"{FINAL_CLASSES.index(cname)} "
                        f"{cx:.6f} {cy:.6f} {sw/W:.6f} {sh/H:.6f}"
                    )
                    success = True
                    break

        # ================= DISTRACTORS (more frequent, variable size) =================

        n_dist = RNG.choices(
            [0, 1, 2, 3],
            weights=[40, 35, 15, 10]   # was [70, 25, 5]
        )[0]

        for _ in range(n_dist):
            if not distractors:
                break

            d = safe_read(RNG.choice(distractors))
            if d is None:
                continue

            # Variable size instead of fixed 50×50
            dist_size = RNG.randint(30, 80)
            d = cv2.resize(d, (dist_size, dist_size))

            dpx = RNG.randint(0, W - dist_size)
            dpy = RNG.randint(0, H - dist_size)

            if np.sum(occupancy_mask[dpy:dpy+dist_size, dpx:dpx+dist_size]) == 0:
                if d.shape[2] == 4:
                    composed = paste(composed, d[:, :, :3], d[:, :, 3], dpx, dpy)
                else:
                    composed[dpy:dpy+dist_size, dpx:dpx+dist_size] = d

        composed = apply_scene_effects(composed)

        path_set = (IMG_TRAIN_DIR, LBL_TRAIN_DIR) if RNG.random() < TRAIN_RATIO \
                   else (IMG_VAL_DIR, LBL_VAL_DIR)

        cv2.imwrite(
            os.path.join(path_set[0], f"{idx}.jpg"),
            composed,
            [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        with open(os.path.join(path_set[1], f"{idx}.txt"), "w") as f:
            f.write("\n".join(labels))

    print("DONE")


if __name__ == "__main__":
    generate()