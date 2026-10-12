import cv2
import numpy as np
import random
import os
import glob

def apply_tv_static(img):
    noise = np.random.randint(0, 256, img.shape, dtype=np.uint8)
    return cv2.addWeighted(img, 0.65, noise, 0.35, 0)

def apply_pixelate(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 15, h // 15), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_strong_blur(img):
    k = 41
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_uneven_illumination(img):
    h, w = img.shape[:2]
    x = np.linspace(0.1, 1.8, w)
    y = np.linspace(0.1, 1.8, h)
    X, Y = np.meshgrid(x, y)
    mask = X * Y
    mask = mask / np.max(mask)
    
    img_float = img.astype(np.float32)
    out = img_float * mask[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_darkness(img):
    factor = random.uniform(0.2, 0.35)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def apply_overexposure(img):
    factor = random.uniform(2.0, 2.5)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def apply_sun_glare(img):
    H, W = img.shape[:2]
    cx, cy = int(W * 0.8), int(H * 0.2)
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (int(W * 0.6), int(H * 0.6)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (201, 201), 0)
    
    glare = np.full_like(img, 255, dtype=np.float32)
    out = np.clip(img.astype(np.float32) * (1 - 0.7 * mask[:, :, None]) + glare * (0.7 * mask[:, :, None]), 0, 255)
    return out.astype(np.uint8)

def apply_heavy_shadow(img):
    H, W = img.shape[:2]
    shadow = img.copy().astype(np.float32)
    shadow[H//3:H, 0:W//2] *= 0.15
    return shadow.astype(np.uint8)

def apply_motion_blur(img):
    k = 35
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = 1
    kernel /= k
    return cv2.filter2D(img, -1, kernel)

def apply_thick_fog(img):
    fog_layer = np.full_like(img, 200)
    return cv2.addWeighted(img, 0.5, fog_layer, 0.5, 0)

def apply_salt_pepper_noise(img, amount=0.08):
    out = img.copy()
    h, w = img.shape[:2]
    num_pixels = int(amount * h * w)
    out[np.random.randint(0, h, num_pixels//2), np.random.randint(0, w, num_pixels//2)] = 255
    out[np.random.randint(0, h, num_pixels//2), np.random.randint(0, w, num_pixels//2)] = 0
    return out

def apply_color_shift(img):
    out = img.astype(np.float32)
    out[:, :, 0] = np.clip(out[:, :, 0] * 1.4, 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] * 0.7, 0, 255)
    return out.astype(np.uint8)

def main():
    input_dir = 'input_bgs'
    output_dir = 'output_augmented'
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.*')))
    
    if not image_paths:
        return

    global_counter = 1

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        effects = [
            img,
            apply_tv_static(img),
            apply_pixelate(img),
            apply_strong_blur(img),
            apply_uneven_illumination(img),
            apply_darkness(img),
            apply_overexposure(img),
            apply_sun_glare(img),
            apply_heavy_shadow(img),
            apply_motion_blur(img),
            apply_thick_fog(img),
            apply_salt_pepper_noise(img),
            apply_color_shift(img)
        ]

        for augmented_img in effects:
            out_name = f"bg{global_counter:02d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, augmented_img)
            global_counter += 1

if __name__ == "__main__":
    main()