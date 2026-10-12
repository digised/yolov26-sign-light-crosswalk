#!/usr/bin/env python3
"""
Convert a YOLO .pt (PyTorch, Ultralytics) model into a .blob file that can
run on Luxonis OAK cameras (DepthAI / Myriad X VPU).

Pipeline:
    .pt  --(ultralytics export)-->  .onnx
    .onnx --(blobconverter, compiles remotely for Myriad X)--> .blob

blobconverter's from_onnx() sends the ONNX file to Luxonis's build service,
which runs the OpenVINO Model Optimizer + Compile Tool for you — so you do
NOT need openvino-dev/mo installed locally, and it works even though that
package is deprecated/removed for OpenVINO 2025+.

Usage:
    python pt_to_blob.py --weights last.pt --imgsz 640 --shaves 6

Requirements (install first):
    pip install ultralytics onnx onnxslim onnxruntime
    pip install blobconverter

Notes:
    - shaves: number of SHAVE cores to compile for (OAK boards typically
      support 4-13; 6 is a common default that balances speed vs accuracy).
    - This targets YOLOv5/v8-style Ultralytics export. If you're using a
      different YOLO variant (e.g. YOLOv7 from its own repo), the ONNX
      export step differs slightly — see the --onnx flag to skip export.
    - If your model is an NMS-baked "end-to-end" export (Ultralytics
      YOLOv8+ with an output shape like (1, 300, 6)), the standard DepthAI
      YoloDetectionNetwork node expects raw grid outputs and will NOT
      decode this correctly out of the box — you'd need a custom parsing
      node, or re-export without end-to-end NMS. Check the printed output
      shape from the export step below.
    - blobconverter's public server can be slow or fail on some YOLOv8+
      exports. If from_onnx() hangs or errors, try Luxonis's actively
      maintained converter instead: https://tools.luxonis.com/
"""

import argparse
import sys
from pathlib import Path


def export_to_onnx(weights: Path, imgsz: int, opset: int) -> Path:
    """Export .pt -> .onnx using the ultralytics package."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed. Run: pip install ultralytics")

    print(f"[1/2] Loading {weights} and exporting to ONNX (imgsz={imgsz}, opset={opset})...")
    model = YOLO(str(weights))
    # end2end=False forces the classic raw-grid head (1, nc+4, 8400) instead
    # of YOLO26's default NMS-free end-to-end head (1, 300, 6). The Myriad X
    # (RVC2) compiler used by OAK-D Lite doesn't support the TopK/Tile ops
    # in the end-to-end head, and DepthAI's YoloDetectionNetwork node is
    # built to decode the classic raw-grid format anyway.
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, end2end=False)
    onnx_path = Path(onnx_path)
    print(f"    -> ONNX saved to {onnx_path}")
    return onnx_path


def onnx_to_blob(onnx_path: Path, shaves: int, output_dir: Path) -> Path:
    """Compile ONNX directly into a Myriad X .blob via blobconverter's remote build service."""
    try:
        import blobconverter
    except ImportError:
        sys.exit("blobconverter is not installed. Run: pip install blobconverter")

    print(f"[2/2] Compiling ONNX to .blob (shaves={shaves})... this calls a remote build service and can take a few minutes.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # scale=255 and RGB order match how Ultralytics normalizes input during
    # training/inference (pixels/255, RGB), so the blob expects the same.
    blob_path = blobconverter.from_onnx(
        model=str(onnx_path),
        data_type="FP16",
        shaves=shaves,
        optimizer_params=[
            "--scale=255",
            "--reverse_input_channels",
        ],
        output_dir=str(output_dir),
    )
    print(f"    -> Blob saved to {blob_path}")
    return Path(blob_path)


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO .pt -> OAK .blob")
    parser.add_argument("--weights", type=Path, required=True, help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (square)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--shaves", type=int, default=6, help="Number of SHAVE cores (4-13)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("oak_export"),
        help="Directory to store ONNX/IR/blob outputs"
    )
    parser.add_argument(
        "--onnx", type=Path, default=None,
        help="Skip the .pt export step and use an existing .onnx file"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.onnx:
        onnx_path = args.onnx
        print(f"Skipping export step, using existing ONNX: {onnx_path}")
    else:
        if not args.weights.exists():
            sys.exit(f"Weights file not found: {args.weights}")
        onnx_path = export_to_onnx(args.weights, args.imgsz, args.opset)

    blob_path = onnx_to_blob(onnx_path, args.shaves, args.output_dir)

    print("\nDone!")
    print(f"Blob file ready for OAK: {blob_path}")
    print("Load it in DepthAI with:")
    print(f'    nn.setBlobPath("{blob_path}")')


if __name__ == "__main__":
    main()