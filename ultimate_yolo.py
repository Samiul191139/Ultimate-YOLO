#!/usr/bin/env python3
"""
Unified YOLO runner (v8..v12 compatible via ultralytics.YOLO)

Usage examples:
  python yolo_all_versions_runner.py --data data.yaml --weights yolov8n.pt --task detect --epochs 50 --imgsz 416 --batch 8 --device cuda --save-rois
  python yolo_all_versions_runner.py --data data.yaml --weights yolov9c.pt --task detect --epochs 50 --device cuda --save-rois

Notes:
- 'val' and 'test' splits are read from data.yaml; validation is for annotated eval.
- ROI saving crops predictions and writes files named <orig>__pred_<class>_<conf>.png
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Unified YOLO runner (v8..v12 via ultralytics.YOLO)")
    # data / model
    p.add_argument("--task", choices=["detect", "classify"], default="detect")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--weights", required=True, help="weights file (.pt) or model yaml (.yaml). e.g. yolov8n.pt, yolov9c.pt, yolov10n.yaml")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr0", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="runs/exp")
    p.add_argument("--project-name", type=str, default="single_run")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold for inference")
    p.add_argument("--save-rois", action="store_true", help="save detected bounding boxes as cropped images during inference")
    p.add_argument("--save-overlays", action="store_true", help="save overlay images comparing GT vs Pred during inference (requires GT labels)")
    p.add_argument("--test-images", type=str, default=None, help="optional directory for qualitative inference (images)")
    p.add_argument("--class-names", type=str, default=None, help="optional JSON mapping from class index to name, e.g. '{\"0\":\"class1\",\"1\":\"class2\"}' etc.")
    p.add_argument("--no-plot", action="store_true", help="skip plotting training curves")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    return p.parse_args()


# ---------------- utilities ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _find_column(df: pd.DataFrame, substring: str):
    for col in df.columns:
        if substring in col:
            return col
    return None


def save_training_curves(csv_path: str, save_dir: Path):
    if not Path(csv_path).exists():
        print("results.csv not found, skipping curves")
        return
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Loss plot
    col_train_box = _find_column(df, "train/box_loss")
    col_val_box = _find_column(df, "val/box_loss")
    if col_train_box and col_val_box:
        axes[0].plot(df[col_train_box], label="Train Box Loss")
        axes[0].plot(df[col_val_box], label="Val Box Loss")
        axes[0].legend()
        axes[0].set_title("Box loss")
    # Metrics plot
    col_prec = _find_column(df, "metrics/precision")
    col_rec = _find_column(df, "metrics/recall")
    col_map = _find_column(df, "metrics/mAP50")
    if col_prec or col_rec or col_map:
        if col_prec:
            axes[1].plot(df[col_prec], label="Precision")
        if col_rec:
            axes[1].plot(df[col_rec], label="Recall")
        if col_map:
            axes[1].plot(df[col_map], label="mAP50")
        axes[1].legend()
        axes[1].set_title("Validation metrics")
    fig.tight_layout()
    out = save_dir / "training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved training curves: {out}")


# ---------------- overlay / box utils ----------------
def yolo_xywh_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2


def overlay_gt_pred(img, gt_boxes, pred_boxes, class_map=None):
    # gt_boxes & pred_boxes are lists of (cls, cx, cy, w, h) normalized
    out = img.copy()
    h, w = out.shape[:2]
    # draw GT (green)
    for cls, cx, cy, bw, bh in gt_boxes:
        x1, y1, x2, y2 = yolo_xywh_to_xyxy(cx, cy, bw, bh, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if class_map:
            txt = class_map.get(int(cls), str(cls))
            cv2.putText(out, txt, (x1, max(y1 - 6, 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    # draw preds (red)
    for cls, cx, cy, bw, bh, conf in pred_boxes:
        x1, y1, x2, y2 = yolo_xywh_to_xyxy(cx, cy, bw, bh, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 1)
        name = class_map.get(int(cls), str(cls)) if class_map else str(cls)
        cv2.putText(out, f"{name} {conf:.2f}", (x1, min(y2 + 12, h - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return out


# ---------------- runner class ----------------
class YOLORunner:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.run_dir = Path(args.output_dir) / args.project_name
        ensure_dir(self.run_dir)
        self.class_map = None
        if args.class_names:
            try:
                self.class_map: Dict[int, str] = json.loads(args.class_names)
            except Exception:
                print("Failed to parse --class-names JSON; expecting mapping like '{\"0\":\"LSS\",\"1\":\"RSS\"}'")
        # set seed if desired (optional)

    def build_model(self):
        print(f"Loading model from: {self.args.weights}")
        self.model = YOLO(self.args.weights, task=self.args.task)
        # print basic model info
        try:
            self.model.info()
        except Exception:
            pass

    def train(self):
        if self.model is None:
            self.build_model()
        print("Starting training...")
        results = self.model.train(
            data=self.args.data,
            epochs=self.args.epochs,
            imgsz=self.args.imgsz,
            batch=self.args.batch,
            lr0=self.args.lr0,
            device=self.args.device,
            project=str(self.run_dir.parent),
            name=self.run_dir.name,
            val=True,
        )
        # returned object contains .save_dir or .save_dir attribute
        try:
            self.run_dir = Path(results.save_dir)
        except Exception:
            # fallback
            pass
        print("Training finished. run_dir:", self.run_dir)
        # save metrics.json if available (results.results_dict)
        try:
            metrics = getattr(results, "results_dict", None)
            if metrics:
                with open(self.run_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
        except Exception:
            pass
        return results

    def validate(self, weights_path: Optional[str] = None):
        if weights_path is None:
            # take latest in run_dir/weights
            weights_path = str((self.run_dir / "weights" / "best.pt"))
        print("Validating using:", weights_path)
        model = YOLO(weights_path, task=self.args.task)
        val_results = model.val(data=self.args.data, split="val", device=self.args.device,
                                project=str(self.run_dir), name="val")
        return val_results

    def test(self, weights_path: Optional[str] = None):
        if weights_path is None:
            weights_path = str((self.run_dir / "weights" / "best.pt"))
        print("Testing using:", weights_path)
        model = YOLO(weights_path, task=self.args.task)
        test_results = model.val(data=self.args.data, split="test", device=self.args.device,
                                 project=str(self.run_dir), name="test")
        return test_results

    def infer_and_save(self, weights_path: Optional[str] = None, source: Optional[str] = None):
        """
        Run inference and optionally save ROIs and overlays.
        """
        if weights_path is None:
            weights_path = str((self.run_dir / "weights" / "best.pt"))
        if source is None:
            source = self.args.test_images
            if source is None:
                raise ValueError("No source specified for inference. Provide --test-images or call with source param.")

        model = YOLO(weights_path, task=self.args.task)
        out_dir = self.run_dir / "inference"
        ensure_dir(out_dir)
        # run prediction; set save=False (we'll control saving)
        preds = model.predict(source=source, conf=self.args.conf, device=self.args.device, save=False, save_txt=True)

        # preds is a list of Results (one per image)
        # Create ROI dir
        rois_dir = out_dir / "rois"
        ensure_dir(rois_dir)
        # overlay dir
        overlay_dir = out_dir / "overlays"
        ensure_dir(overlay_dir)

        # iterate results
        for res in preds:
            # res.path is usually original path (string) for that image
            img_path = None
            try:
                img_path = res.path
            except Exception:
                # fallback
                img_path = None

            # prefer reading from res.orig_img if provided
            img_np = None
            if hasattr(res, "orig_img") and getattr(res, "orig_img") is not None:
                # ultralytics stores numpy BGR in res.orig_img
                img_np = getattr(res, "orig_img")
            elif img_path:
                img_np = cv2.imread(str(img_path))
            else:
                print("No image available for a result; skipping")
                continue

            h, w = img_np.shape[:2]
            boxes = []
            # extract predicted boxes; ultralytics Results.boxes has .xyxy, .cls, .conf or .xywhn
            try:
                boxes_tensor = res.boxes  # contains attributes
                # xyxy (pixel) or xywhn (normalized)
                if hasattr(boxes_tensor, "xyxy"):
                    xyxy = boxes_tensor.xyxy.cpu().numpy()  # (N,4)
                    confs = boxes_tensor.conf.cpu().numpy() if hasattr(boxes_tensor, "conf") else np.ones(len(xyxy))
                    cls_ids = boxes_tensor.cls.cpu().numpy().astype(int) if hasattr(boxes_tensor, "cls") else np.zeros(len(xyxy), dtype=int)
                    for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
                        # store as (cid, cx_norm, cy_norm, w_norm, h_norm, conf) for compatibility
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        boxes.append((int(cid), float(cx), float(cy), float(bw), float(bh), float(c)))
                else:
                    # use normalized xywh if available
                    if hasattr(boxes_tensor, "xywhn"):
                        xywhn = boxes_tensor.xywhn.cpu().numpy()
                        confs = boxes_tensor.conf.cpu().numpy() if hasattr(boxes_tensor, "conf") else np.ones(len(xywhn))
                        cls_ids = boxes_tensor.cls.cpu().numpy().astype(int) if hasattr(boxes_tensor, "cls") else np.zeros(len(xywhn), dtype=int)
                        for (cx, cy, bw, bh), c, cid in zip(xywhn, confs, cls_ids):
                            boxes.append((int(cid), float(cx), float(cy), float(bw), float(bh), float(c)))
            except Exception as e:
                print("Failed to parse boxes for result:", e)
                continue

            # image filename (basename)
            img_name = Path(res.path).name if res.path else f"img_{np.random.randint(1e9)}.png"

            # save ROIs as crops
            if self.args.save_rois and boxes:
                for i, (cid, cx, cy, bw, bh, conf) in enumerate(boxes):
                    x1, y1, x2, y2 = yolo_xywh_to_xyxy(cx, cy, bw, bh, w, h)
                    crop = img_np[y1:y2, x1:x2].copy()
                    cls_name = self.class_map.get(cid, str(cid)) if self.class_map else str(cid)
                    # severity not available here unless you encode it into original filename; we keep class+conf for name
                    crop_name = f"{Path(img_name).stem}__pred__{cls_name}__{conf:.3f}__{i}.png"
                    crop_path = rois_dir / crop_name
                    cv2.imwrite(str(crop_path), crop)

            # create overlay if required and if gt labels exist
            if self.args.save_overlays:
                # Attempt to find GT label in the same folder as image (replace images -> labels, or .png->.txt)
                gt_label_path = None
                if res.path:
                    # try same folder with .txt
                    alt = Path(res.path).with_suffix(".txt")
                    if alt.exists():
                        gt_label_path = alt
                    else:
                        # try swapping 'images' with 'labels' in path
                        pstr = str(res.path)
                        if "/images/" in pstr:
                            alt2 = pstr.replace("/images/", "/labels/")
                            if Path(alt2).exists():
                                gt_label_path = Path(alt2)
                gt_boxes = []
                if gt_label_path and gt_label_path.exists():
                    with open(gt_label_path, "r") as f:
                        for ln in f:
                            parts = ln.strip().split()
                            if len(parts) >= 5:
                                _cls, cx, cy, bw, bh = parts[:5]
                                gt_boxes.append((int(_cls), float(cx), float(cy), float(bw), float(bh)))
                pred_boxes_simple = [(cid, cx, cy, bw, bh, conf) for (cid, cx, cy, bw, bh, conf) in boxes]
                overlay_img = overlay_gt_pred(img_np, gt_boxes, pred_boxes_simple, class_map=self.class_map)
                overlay_path = overlay_dir / img_name
                cv2.imwrite(str(overlay_path), overlay_img)

        print("Inference and ROI/overlay saving complete. Saved to:", out_dir)
        return out_dir


# ---------------- main ----------------
def main():
    args = parse_args()
    runner = YOLORunner(args)
    runner.build_model()

    # Train
    results = runner.train()

    # Validate
    run_dir = runner.run_dir
    best_wt = run_dir / "weights" / "best.pt"
    if best_wt.exists():
        val_res = runner.validate(weights_path=str(best_wt))
        print("Validation done.")
    else:
        print("No best.pt found after training in", run_dir)

    # Test split (if present in data.yaml)
    test_res = None
    try:
        test_res = runner.test(weights_path=str(best_wt))
        print("Test done.")
    except Exception as e:
        print("Test failed or test split not defined:", e)

    # Save curves/metrics
    results_csv = run_dir / "results.csv"
    if results_csv.exists() and not args.no_plot:
        save_training_curves(str(results_csv), run_dir)

    # Inference + ROIs
    if args.test_images:
        out = runner.infer_and_save(weights_path=str(best_wt), source=args.test_images)
        print("Inference outputs:", out)

    print("All finished. run_dir:", run_dir)


if __name__ == "__main__":
    main()