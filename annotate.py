#!/usr/bin/env python3
"""
annotate.py -- Manual bounding-box annotation tool for traffic-light labeling
==============================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from PIL import Image, ImageTk

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_CLASSES_FILE = "classes.txt"

# این لیست دقیقا با FINAL_CLASSES در generate_dataset.py همگام شده است
DEFAULT_CLASSES = [
    "barred", "crossWalk", "NoPassB", "NoPassE", "park", "priority",
    "stop", "straight", "tunnelB", "tunnelE", "turn", "slope",
    "school", "speed", "CrossWalk", "Red", "Green",
]

CONFIG_PATH = Path.home() / ".traffic_light_annotator_config.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

COLOR_PALETTE = [
    "#e74c3c", "#2ecc71", "#f1c40f", "#3498db", "#9b59b6", "#e67e22",
    "#1abc9c", "#95a5a6", "#34495e", "#d35400", "#16a085", "#2980b9",
    "#8e44ad", "#c0392b", "#27ae60", "#f39c12", "#7f8c8d",
]

HANDLE_SIZE = 8          # px
MIN_BOX_SIZE = 15        # افزایش یافت تا از ایجاد کادرهای تصادفی با یک کلیک ساده جلوگیری شود
HANDLE_NAMES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"]

# --------------------------------------------------------------------------
# UI Colors (Dark Theme)
# --------------------------------------------------------------------------
BG_MAIN = "#2b2b2b"
BG_SIDE = "#3c3f41"
BG_CANVAS = "#1e1e1e"
FG_TEXT = "#ffffff"
BTN_BG = "#5c5c5c"
BTN_HOVER = "#757575"

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

class Box:
    __slots__ = ("class_id", "x1", "y1", "x2", "y2")

    def __init__(self, class_id, x1, y1, x2, y2):
        self.class_id = class_id
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def normalize(self):
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

    def clamp(self, W, H):
        self.x1 = max(0, min(self.x1, W))
        self.x2 = max(0, min(self.x2, W))
        self.y1 = max(0, min(self.y1, H))
        self.y2 = max(0, min(self.y2, H))

    def is_degenerate(self):
        return (self.x2 - self.x1) < MIN_BOX_SIZE or (self.y2 - self.y1) < MIN_BOX_SIZE

    def to_yolo_line(self, W, H):
        self.normalize()
        cx = (self.x1 + self.x2) / 2.0 / W
        cy = (self.y1 + self.y2) / 2.0 / H
        sw = (self.x2 - self.x1)
        sh = (self.y2 - self.y1)
        return f"{self.class_id} {cx:.6f} {cy:.6f} {sw / W:.6f} {sh / H:.6f}"

    @staticmethod
    def from_yolo_line(line, W, H):
        parts = line.split()
        if len(parts) != 5:
            return None
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(p) for p in parts[1:])
        cx *= W; cy *= H; w *= W; h *= H
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return Box(class_id, x1, y1, x2, y2)

class ImageEntry:
    def __init__(self, path: Path, label_path: Path):
        self.path = path
        self.label_path = label_path
        self.boxes = []
        self.marked_empty = False
        self.loaded_from_disk = False

    def status_char(self):
        if self.marked_empty: return "E"
        if self.boxes: return "L"
        if self.label_path.exists(): return "L"
        return " "

# --------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------

def load_classes(path: Path):
    path.write_text("\n".join(DEFAULT_CLASSES) + "\n", encoding="utf-8")
    return DEFAULT_CLASSES

def color_for_class(class_id):
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]

def resolve_label_path(image_path: Path, mode: str) -> Path:
    labels_dir = image_path.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    return labels_dir / image_path.with_suffix(".txt").name

def load_config():
    if CONFIG_PATH.exists():
        try: return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

def save_config(cfg):
    try: CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e: print(f"[warn] could not write config: {e}")

# --------------------------------------------------------------------------
# Flat Button Helper
# --------------------------------------------------------------------------
def create_flat_button(parent, text, command, bg=BTN_BG):
    btn = tk.Button(parent, text=text, command=command, bg=bg, fg=FG_TEXT, 
                    activebackground=BTN_HOVER, activeforeground=FG_TEXT, 
                    relief=tk.FLAT, padx=10, pady=4, font=("Segoe UI", 9))
    return btn

# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------

class AnnotatorApp:
    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.args = args
        self.classes = load_classes(Path(args.classes))
        self.labels_mode = args.labels_dir

        self.entries = []
        self.index = 0
        self.pil_image = None
        self.tk_image = None
        self.img_w, self.img_h = 0, 0
        self.scale, self.fit_scale, self.pan_x, self.pan_y = 1.0, 1.0, 0, 0
        self.selected_box = None
        self.last_used_class = 0
        self.session_counts = {c: 0 for c in range(len(self.classes))}

        self.drag_mode = None
        self.drag_start_screen = None
        self.drag_start_image = None
        self.drag_box_orig = None
        self.drag_handle = None
        self.pending_box = None
        self.space_held = False
        self.class_popup = None

        self._build_ui()
        self._bind_keys()
        self._load_image_set(args)
        if self.entries:
            self._go_to_index(self._initial_index())

    def _build_ui(self):
        self.root.title("Traffic Light Annotator - Dark Mode")
        self.root.geometry("1200x800")
        self.root.configure(bg=BG_MAIN)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        main = tk.Frame(self.root, bg=BG_MAIN)
        main.pack(fill=tk.BOTH, expand=True)

        canvas_frame = tk.Frame(main, bg=BG_CANVAS)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg=BG_CANVAS, highlightthickness=0, cursor="tcross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_down)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)

        side = tk.Frame(main, width=260, bg=BG_SIDE)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        tk.Label(side, text="Classes (legend)", bg=BG_SIDE, fg=FG_TEXT, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(8, 2))
        
        legend_container = tk.Frame(side, bg=BG_SIDE)
        legend_container.pack(fill=tk.X, padx=8)
        legend_canvas = tk.Canvas(legend_container, bg=BG_SIDE, highlightthickness=0, height=180)
        legend_scroll = tk.Scrollbar(legend_container, orient=tk.VERTICAL, command=legend_canvas.yview)
        self.legend_frame = tk.Frame(legend_canvas, bg=BG_SIDE)
        self.legend_frame.bind("<Configure>", lambda e: legend_canvas.configure(scrollregion=legend_canvas.bbox("all")))
        legend_canvas.create_window((0, 0), window=self.legend_frame, anchor="nw")
        legend_canvas.configure(yscrollcommand=legend_scroll.set)
        legend_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        legend_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._build_legend()

        tk.Label(side, text="Session counts", bg=BG_SIDE, fg=FG_TEXT, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(12, 2))
        self.counts_label = tk.Label(side, text="", bg=BG_SIDE, fg="#cccccc", justify=tk.LEFT, anchor="w")
        self.counts_label.pack(fill=tk.X, padx=8)

        tk.Label(side, text="Images", bg=BG_SIDE, fg=FG_TEXT, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(12, 2))
        list_frame = tk.Frame(side, bg=BG_SIDE)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, bg=BG_MAIN, fg=FG_TEXT,
                                  selectbackground="#3498db", font=("Consolas", 9), activestyle="none", highlightthickness=0)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        bottom = tk.Frame(self.root, bg=BG_SIDE)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        create_flat_button(bottom, "< Prev (A)", self.prev_image).pack(side=tk.LEFT, padx=4, pady=8)
        create_flat_button(bottom, "Next (D) >", self.next_image).pack(side=tk.LEFT, padx=4, pady=8)
        create_flat_button(bottom, "Save (S)", lambda: self.save_current(manual=True)).pack(side=tk.LEFT, padx=4, pady=8)
        create_flat_button(bottom, "No objects (N)", self.mark_empty).pack(side=tk.LEFT, padx=4, pady=8)
        create_flat_button(bottom, "Undo (Ctrl+Z)", self.undo_last).pack(side=tk.LEFT, padx=4, pady=8)
        create_flat_button(bottom, "Open folder...", self._open_folder_dialog, bg="#2980b9").pack(side=tk.LEFT, padx=12, pady=8)

        status_frame = tk.Frame(bottom, bg=BG_SIDE)
        status_frame.pack(side=tk.RIGHT, padx=8)
        
        tk.Label(status_frame, text="💡 Tip: Right-click or Esc while dragging to Cancel", bg=BG_SIDE, fg="#f39c12", font=("Segoe UI", 8, "italic")).pack(anchor="e")
        self.status_label = tk.Label(status_frame, text="", bg=BG_SIDE, fg=FG_TEXT, anchor="e")
        self.status_label.pack(anchor="e")

    def _build_legend(self):
        for child in self.legend_frame.winfo_children(): child.destroy()
        for cid, name in enumerate(self.classes):
            row = tk.Frame(self.legend_frame, bg=BG_SIDE)
            row.pack(fill=tk.X, pady=1)
            swatch = tk.Canvas(row, width=14, height=14, bg=color_for_class(cid), highlightthickness=0)
            swatch.pack(side=tk.LEFT, padx=(0, 6))
            tk.Label(row, text=f"[{cid}] {name}", bg=BG_SIDE, fg=FG_TEXT, anchor="w").pack(side=tk.LEFT)

    def _initial_index(self):
        cfg = load_config()
        folder_key = str(self._session_folder())
        last = cfg.get(folder_key)
        if last is not None and 0 <= last < len(self.entries): return last
        return 0

    def _session_folder(self):
        if self.entries: return self.entries[0].path.parent
        return Path(".")

    def _load_image_set(self, args):
        if args.image:
            p = Path(args.image).resolve()
            if not p.exists(): raise SystemExit(f"Image not found: {p}")
            paths = [p]
        elif args.folder:
            folder = Path(args.folder).resolve()
            if not folder.is_dir(): raise SystemExit(f"Folder not found: {folder}")
            paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)
        else:
            paths = self._prompt_for_input()

        self.entries = [ImageEntry(p, resolve_label_path(p, self.labels_mode)) for p in paths]
        if not self.entries: messagebox.showwarning("No images", "No images were found/selected.")
        self._refresh_listbox()

    def _prompt_for_input(self):
        choice = messagebox.askyesno("Open images", "Open a FOLDER of images?\n\nYes = choose a folder\nNo = choose a single image file")
        if choice:
            folder = filedialog.askdirectory(title="Select image folder")
            if not folder: return []
            return sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)
        else:
            f = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.jpg *.jpeg *.png")])
            return [Path(f)] if f else []

    def _open_folder_dialog(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder: return
        self.save_current()
        paths = sorted([p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS], key=lambda p: p.name)
        self.entries = [ImageEntry(p, resolve_label_path(p, self.labels_mode)) for p in paths]
        self._refresh_listbox()
        if self.entries: self._go_to_index(0)

    def _go_to_index(self, idx):
        if not self.entries: return
        idx = max(0, min(idx, len(self.entries) - 1))
        self.index = idx
        entry = self.entries[idx]
        self._load_entry_image(entry)
        self._load_entry_labels(entry)
        self.selected_box = None
        self._fit_to_window()
        self._render()
        self._refresh_listbox()
        self._update_status()
        self._persist_session()

    def next_image(self):
        self.save_current()
        self._go_to_index(self.index + 1)

    def prev_image(self):
        self.save_current()
        self._go_to_index(self.index - 1)

    def _on_listbox_select(self, _evt):
        sel = self.listbox.curselection()
        if sel:
            self.save_current()
            self._go_to_index(sel[0])

    def _load_entry_image(self, entry: ImageEntry):
        img = Image.open(entry.path).convert("RGB")
        self.pil_image = img
        self.img_w, self.img_h = img.size

    def _load_entry_labels(self, entry: ImageEntry):
        if entry.loaded_from_disk: return
        entry.loaded_from_disk = True
        if entry.label_path.exists():
            text = entry.label_path.read_text(encoding="utf-8").strip()
            if text == "":
                entry.marked_empty = True
                entry.boxes = []
            else:
                boxes = []
                for line in text.splitlines():
                    b = Box.from_yolo_line(line.strip(), self.img_w, self.img_h)
                    if b is not None: boxes.append(b)
                entry.boxes = boxes
                entry.marked_empty = False

    def save_current(self, manual=False):
        if not self.entries: return
        entry = self.entries[self.index]
        if not entry.boxes:
            entry.label_path.write_text("", encoding="utf-8")
            entry.marked_empty = True
        else:
            lines = [b.to_yolo_line(self.img_w, self.img_h) for b in entry.boxes]
            entry.label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            entry.marked_empty = False
        self._refresh_listbox()
        self._update_status()

    def mark_empty(self):
        if not self.entries: return
        entry = self.entries[self.index]
        entry.boxes = []
        entry.marked_empty = True
        self.selected_box = None
        self._render()
        self.save_current()

    def _fit_to_window(self):
        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        self.fit_scale = 1.0 if (self.img_w == 0 or self.img_h == 0) else min(cw / self.img_w, ch / self.img_h)
        self.scale = self.fit_scale
        self.pan_x = (cw - self.img_w * self.scale) / 2
        self.pan_y = (ch - self.img_h * self.scale) / 2

    def _on_canvas_resize(self, _evt):
        if self.pil_image is not None:
            self._fit_to_window()
            self._render()

    def image_to_screen(self, ix, iy):
        return ix * self.scale + self.pan_x, iy * self.scale + self.pan_y

    def screen_to_image(self, sx, sy):
        return (sx - self.pan_x) / self.scale, (sy - self.pan_y) / self.scale

    def _zoom(self, factor, center_screen=None):
        if self.pil_image is None: return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if center_screen is None: center_screen = (cw / 2, ch / 2)
        cx_img, cy_img = self.screen_to_image(*center_screen)
        self.scale = max(self.fit_scale * 0.1, min(self.scale * factor, self.fit_scale * 40))
        self.pan_x = center_screen[0] - cx_img * self.scale
        self.pan_y = center_screen[1] - cy_img * self.scale
        self._render()

    def _on_wheel(self, event):
        if event.num == 5 or getattr(event, "delta", 0) < 0: self._zoom(0.9, (event.x, event.y))
        else: self._zoom(1.1, (event.x, event.y))

    def _render(self):
        if self.pil_image is None:
            self.canvas.delete("all")
            return
        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        vis_x1, vis_y1 = self.screen_to_image(0, 0)
        vis_x2, vis_y2 = self.screen_to_image(cw, ch)
        vis_x1, vis_y1 = max(0, int(vis_x1)), max(0, int(vis_y1))
        vis_x2, vis_y2 = min(self.img_w, int(vis_x2) + 1), min(self.img_h, int(vis_y2) + 1)

        self.canvas.delete("all")
        if vis_x2 > vis_x1 and vis_y2 > vis_y1:
            crop = self.pil_image.crop((vis_x1, vis_y1, vis_x2, vis_y2))
            disp_w, disp_h = max(1, int((vis_x2 - vis_x1) * self.scale)), max(1, int((vis_y2 - vis_y1) * self.scale))
            resample = Image.BILINEAR if self.scale < 4 else Image.NEAREST
            self.tk_image = ImageTk.PhotoImage(crop.resize((disp_w, disp_h), resample))
            sx, sy = self.image_to_screen(vis_x1, vis_y1)
            self.canvas.create_image(sx, sy, anchor="nw", image=self.tk_image)

        for box in self.entries[self.index].boxes:
            self._draw_box(box, selected=(box is self.selected_box))
        if self.pending_box is not None:
            self._draw_box(self.pending_box, selected=True, dashed=True)

    def _draw_box(self, box: Box, selected=False, dashed=False):
        x1, y1 = self.image_to_screen(box.x1, box.y1)
        x2, y2 = self.image_to_screen(box.x2, box.y2)
        color = color_for_class(box.class_id)
        width, dash = (3 if selected else 2), ((4, 3) if dashed else None)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, dash=dash)
        name = self.classes[box.class_id] if box.class_id < len(self.classes) else "?"
        tx, ty = x1 + 2, y1 - 14 if y1 - 14 > 0 else y1 + 2
        text_id = self.canvas.create_text(tx, ty, text=name, anchor="nw", fill="white", font=("Segoe UI", 9, "bold"))
        bbox = self.canvas.bbox(text_id)
        if bbox: self.canvas.tag_lower(self.canvas.create_rectangle(bbox, fill=color, outline=color), text_id)
        if selected:
            for name_h, (hx, hy) in self._handle_positions(box).items():
                self.canvas.create_rectangle(hx - HANDLE_SIZE/2, hy - HANDLE_SIZE/2, hx + HANDLE_SIZE/2, hy + HANDLE_SIZE/2, fill="white", outline=color)

    def _handle_positions(self, box: Box):
        x1, y1 = self.image_to_screen(box.x1, box.y1)
        x2, y2 = self.image_to_screen(box.x2, box.y2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        return {"nw": (x1, y1), "n": (mx, y1), "ne": (x2, y1), "e": (x2, my), "se": (x2, y2), "s": (mx, y2), "sw": (x1, y2), "w": (x1, my)}

    def _hit_test_handle(self, sx, sy):
        if self.selected_box is None: return None
        for name, (hx, hy) in self._handle_positions(self.selected_box).items():
            if abs(sx - hx) <= HANDLE_SIZE and abs(sy - hy) <= HANDLE_SIZE: return name
        return None

    def _hit_test_box(self, ix, iy):
        for box in reversed(self.entries[self.index].boxes):
            if box.x1 <= ix <= box.x2 and box.y1 <= iy <= box.y2: return box
        return None

    def _on_left_down(self, event):
        if self.class_popup is not None: return
        if self.space_held:
            self._on_pan_down(event)
            return

        handle = self._hit_test_handle(event.x, event.y)
        if handle is not None:
            self.drag_mode, self.drag_handle = "resize", handle
            self.drag_box_orig = (self.selected_box.x1, self.selected_box.y1, self.selected_box.x2, self.selected_box.y2)
            self.drag_start_image = self.screen_to_image(event.x, event.y)
            return

        ix, iy = self.screen_to_image(event.x, event.y)
        box = self._hit_test_box(ix, iy)
        if box is not None:
            self.selected_box, self.drag_mode = box, "move"
            self.drag_box_orig = (box.x1, box.y1, box.x2, box.y2)
            self.drag_start_image = (ix, iy)
            self._render()
            return

        self.selected_box, self.drag_mode = None, "draw"
        self.drag_start_image = (ix, iy)
        self.pending_box = Box(self.last_used_class, ix, iy, ix, iy)
        self._render()

    def _on_left_drag(self, event):
        if self.drag_mode is None: return
        ix, iy = self.screen_to_image(event.x, event.y)

        if self.drag_mode == "draw" and self.pending_box is not None:
            self.pending_box.x2, self.pending_box.y2 = ix, iy
            self._render()
        elif self.drag_mode == "move" and self.selected_box is not None:
            ox1, oy1, ox2, oy2 = self.drag_box_orig
            dx, dy = ix - self.drag_start_image[0], iy - self.drag_start_image[1]
            w, h = ox2 - ox1, oy2 - oy1
            nx1, ny1 = max(0, min(ox1 + dx, self.img_w - w)), max(0, min(oy1 + dy, self.img_h - h))
            self.selected_box.x1, self.selected_box.y1, self.selected_box.x2, self.selected_box.y2 = nx1, ny1, nx1 + w, ny1 + h
            self._render()
        elif self.drag_mode == "resize" and self.selected_box is not None:
            nx1, ny1, nx2, ny2 = self.drag_box_orig
            h = self.drag_handle
            if "n" in h: ny1 = iy
            if "s" in h: ny2 = iy
            if "w" in h: nx1 = ix
            if "e" in h: nx2 = ix
            self.selected_box.x1, self.selected_box.y1, self.selected_box.x2, self.selected_box.y2 = nx1, ny1, nx2, ny2
            self.selected_box.clamp(self.img_w, self.img_h)
            self._render()

    def _on_left_up(self, event):
        if self.drag_mode == "draw" and self.pending_box is not None:
            self.pending_box.normalize()
            self.pending_box.clamp(self.img_w, self.img_h)
            if self.pending_box.is_degenerate():
                self.pending_box = None
            else:
                self._show_class_selector(event.x, event.y)
        elif self.drag_mode in ("move", "resize") and self.selected_box is not None:
            self.selected_box.normalize()
            self.selected_box.clamp(self.img_w, self.img_h)
        self.drag_mode, self.drag_handle = None, None
        self._render()

    def _on_pan_down(self, event):
        self.drag_mode, self.drag_start_screen, self._pan_origin = "pan", (event.x, event.y), (self.pan_x, self.pan_y)
        self.canvas.config(cursor="fleur")

    def _on_pan_drag(self, event):
        if self.drag_mode != "pan": return
        self.pan_x = self._pan_origin[0] + (event.x - self.drag_start_screen[0])
        self.pan_y = self._pan_origin[1] + (event.y - self.drag_start_screen[1])
        self._render()

    def _on_pan_up(self, _event):
        if self.drag_mode == "pan":
            self.drag_mode = None
            self.canvas.config(cursor="tcross")

    def _on_right_click(self, event):
        if self.drag_mode == "draw":
            self._cancel_current_draw()
            return

        ix, iy = self.screen_to_image(event.x, event.y)
        box = self._hit_test_box(ix, iy)
        if box is None: return
        self.selected_box = box
        self._render()
        menu = tk.Menu(self.root, tearoff=0, bg=BG_SIDE, fg=FG_TEXT)
        menu.add_command(label=f"Change class (current: {self.classes[box.class_id]})", command=lambda: self._show_class_selector(event.x, event.y, target_box=box))
        menu.add_command(label="Delete box", command=self.delete_selected)
        menu.tk_popup(event.x_root, event.y_root)

    def _cancel_current_draw(self):
        if self.class_popup is not None:
            self.class_popup.grab_release()
            self.class_popup.destroy()
            self.class_popup = None
        self.pending_box, self.drag_mode, self.drag_handle = None, None, None
        self._render()

    def _on_escape(self, _event):
        if self.drag_mode == "draw" or self.pending_box is not None or self.class_popup is not None:
            self._cancel_current_draw()

    def _show_class_selector(self, screen_x, screen_y, target_box=None):
        if self.class_popup is not None: self.class_popup.destroy()

        popup = tk.Toplevel(self.root)
        popup.withdraw() # مخفی نگه داشتن موقت برای جلوگیری از پرش تصویر
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)
        self.class_popup = popup

        frame = tk.Frame(popup, bg=BG_MAIN, bd=2, relief=tk.RAISED)
        frame.pack()

        default_id = target_box.class_id if target_box is not None else self.last_used_class
        used_letters = set()

        def pick(cid): self._finalize_class_choice(cid, target_box)

        for cid, name in enumerate(self.classes):
            letter = next((ch for ch in name.lower() if ch.isalpha() and ch not in used_letters), None)
            if letter: used_letters.add(letter)
            shortcut = f"{cid}" if cid <= 9 else ""
            if letter: shortcut = f"{shortcut}/{letter}" if shortcut else letter
            label = f"[{shortcut}] {name}" if shortcut else name
            
            btn = tk.Button(frame, text=label, anchor="w", bg=color_for_class(cid), fg="white", relief=tk.FLAT, command=lambda c=cid: pick(c))
            if cid == default_id: btn.config(relief=tk.SOLID, bd=2)
            btn.pack(fill=tk.X, ipady=3)

        btn_cancel = tk.Button(frame, text="❌ Cancel", anchor="center", bg="#c0392b", fg="white", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), command=lambda: self._cancel_class_choice(target_box))
        btn_cancel.pack(fill=tk.X, ipady=4, pady=(4, 0))

        # ---- محاسبه موقعیت صحیح و جلوگیری از خروج لیست از صفحه ----
        popup.update_idletasks() # به دست آوردن سایز نهایی کادر لیست
        pw = popup.winfo_reqwidth()
        ph = popup.winfo_reqheight()
        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()
        
        root_x = self.canvas.winfo_rootx() + screen_x + 8
        root_y = self.canvas.winfo_rooty() + screen_y + 8
        
        if root_x + pw > sw: root_x = sw - pw - 8
        if root_y + ph > sh: root_y = sh - ph - 8
        root_x = max(0, root_x)
        root_y = max(0, root_y)
        
        popup.geometry(f"+{int(root_x)}+{int(root_y)}")
        popup.deiconify() # نمایش دادن بعد از اصلاح مکان

        # ---- منطق بستن با کلیک بیرون صفحه ----
        def on_click_outside(event):
            # بررسی اینکه آیا کلیک خارج از محدوده پنجره پاپ‌آپ انجام شده است یا نه
            x1, y1 = popup.winfo_rootx(), popup.winfo_rooty()
            x2, y2 = x1 + popup.winfo_width(), y1 + popup.winfo_height()
            if not (x1 <= event.x_root <= x2 and y1 <= event.y_root <= y2):
                self._cancel_class_choice(target_box)

        popup.bind("<Button-1>", on_click_outside)
        popup.bind("<Escape>", lambda e: self._cancel_class_choice(target_box))
        popup.bind("<Return>", lambda e: pick(default_id))

        def make_key_handler():
            letter_map = {}
            used = set()
            for cid, name in enumerate(self.classes):
                for ch in name.lower():
                    if ch.isalpha() and ch not in used:
                        letter_map[ch] = cid
                        used.add(ch)
                        break
            def handler(evt):
                ch = evt.char.lower()
                if ch.isdigit() and int(ch) < len(self.classes): pick(int(ch))
                elif ch in letter_map: pick(letter_map[ch])
            return handler

        popup.bind("<Key>", make_key_handler())
        popup.focus_force()
        popup.grab_set()

    def _finalize_class_choice(self, class_id, target_box):
        if self.class_popup is not None:
            self.class_popup.grab_release()
            self.class_popup.destroy()
            self.class_popup = None
        self.last_used_class = class_id
        if target_box is not None:
            target_box.class_id = class_id
        else:
            if self.pending_box is not None:
                self.pending_box.class_id = class_id
                self.entries[self.index].boxes.append(self.pending_box)
                self.entries[self.index].marked_empty = False
                self.selected_box, self.pending_box = self.pending_box, None
                self.session_counts[class_id] = self.session_counts.get(class_id, 0) + 1
        self._render()
        self._update_status()

    def _cancel_class_choice(self, target_box):
        if self.class_popup is not None:
            self.class_popup.grab_release()
            self.class_popup.destroy()
            self.class_popup = None
        if target_box is None: self.pending_box = None
        self._render()

    def delete_selected(self):
        if self.selected_box is None: return
        entry = self.entries[self.index]
        if self.selected_box in entry.boxes: entry.boxes.remove(self.selected_box)
        self.selected_box = None
        self._render()
        self._update_status()

    def undo_last(self):
        entry = self.entries[self.index]
        if entry.boxes:
            if self.selected_box is entry.boxes.pop(): self.selected_box = None
            self._render()
            self._update_status()

    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, entry in enumerate(self.entries):
            self.listbox.insert(tk.END, f"[{entry.status_char()}] {entry.path.name}")
        if self.entries:
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.index)
            self.listbox.see(self.index)

    def _update_status(self):
        if not self.entries:
            self.status_label.config(text="No images loaded")
            return
        entry = self.entries[self.index]
        counts_str = ", ".join(f"{k}: {v}" for k, v in {self.classes[b.class_id] if b.class_id < len(self.classes) else "?": sum(1 for x in entry.boxes if x.class_id == b.class_id) for b in entry.boxes}.items()) or "0 boxes"
        self.status_label.config(text=f"Image {self.index + 1} / {len(self.entries)}   {entry.path.name}   [{counts_str}]{'  (marked empty)' if entry.marked_empty else ''}")
        self.counts_label.config(text="\n".join(f"{self.classes[cid]}: {self.session_counts.get(cid, 0)}" for cid in range(len(self.classes))))

    def _persist_session(self):
        if not self.entries: return
        cfg = load_config()
        cfg[str(self._session_folder())] = self.index
        save_config(cfg)

    def _bind_keys(self):
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<d>", lambda e: self.next_image()); self.root.bind("<D>", lambda e: self.next_image())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<a>", lambda e: self.prev_image()); self.root.bind("<A>", lambda e: self.prev_image())
        self.root.bind("<Delete>", lambda e: self.delete_selected()); self.root.bind("<BackSpace>", lambda e: self.delete_selected())
        self.root.bind("<Control-z>", lambda e: self.undo_last()); self.root.bind("<Control-Z>", lambda e: self.undo_last())
        self.root.bind("<s>", lambda e: self.save_current(manual=True)); self.root.bind("<S>", lambda e: self.save_current(manual=True))
        self.root.bind("<n>", lambda e: self.mark_empty()); self.root.bind("<N>", lambda e: self.mark_empty())
        self.root.bind("<plus>", lambda e: self._zoom(1.25)); self.root.bind("<equal>", lambda e: self._zoom(1.25))
        self.root.bind("<minus>", lambda e: self._zoom(0.8))
        self.root.bind("<KeyPress-space>", self._on_space_down); self.root.bind("<KeyRelease-space>", self._on_space_up)
        self.root.bind("<Escape>", self._on_escape)

    def _on_space_down(self, _evt):
        self.space_held = True
        self.canvas.config(cursor="fleur")

    def _on_space_up(self, _evt):
        self.space_held = False
        self.canvas.config(cursor="tcross")

    def _on_close(self):
        self.save_current()
        self._persist_session()
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser(description="Manual bounding-box annotation tool for traffic-light labeling.")
    parser.add_argument("--folder", type=str, default=None, help="Folder of images to annotate (jpg/jpeg/png).")
    parser.add_argument("--image", type=str, default=None, help="A single image file to annotate.")
    parser.add_argument("--classes", type=str, default=DEFAULT_CLASSES_FILE, help="Path to classes.txt")
    parser.add_argument("--labels-dir", choices=["same", "parallel"], default="same", help="Label output structure")
    args = parser.parse_args()

    root = tk.Tk()
    app = AnnotatorApp(root, args)
    root.mainloop()

if __name__ == "__main__":
    main()