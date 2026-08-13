"""Compare OCR configurations on real crops rotated through four directions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.components.ocr.paddle_ocr import PaddleOcr
from core.components.ocr.paddleocr.utils import get_rotate_crop_image
from core.config_loader import load_config


ANGLES = (0, 90, 180, 270)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--config", default="config/ocr/paddle.yaml")
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--v6-no-orientation-config", default=None)
    parser.add_argument(
        "--class-registry",
        default="config/class_registry/default.yaml",
    )
    parser.add_argument("--output-dir", default="runs/ocr_rotation")
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def imread(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    encoded.tofile(path)


def rotate_input(rgb: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return rgb
    if angle == 90:
        return cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(rgb, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(angle)


def lcs_recall(gt: str, predicted: str) -> float:
    lengths = [[0] * (len(predicted) + 1) for _ in range(len(gt) + 1)]
    for i, expected in enumerate(gt, start=1):
        for j, actual in enumerate(predicted, start=1):
            lengths[i][j] = (
                lengths[i - 1][j - 1] + 1
                if expected == actual
                else max(lengths[i - 1][j], lengths[i][j - 1])
            )
    return lengths[-1][-1] / len(gt) if gt else 1.0


def load_manifest(input_dir: Path, manifest_path: Path):
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    samples = []
    for row in rows:
        image_path = input_dir / row["image"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        samples.append(
            {
                "image": image_path,
                "gt_text": row["gt_text"],
                "gt_class": row["gt_class"],
            }
        )
    return samples


def load_ocr(config_path: Path, registry_path: Path) -> PaddleOcr:
    ocr_config = load_config(str(config_path))["ocr"]
    registry = load_config(str(registry_path))["class_registry"]
    return PaddleOcr(ocr_config, registry)


def save_debug(output_dir: Path, row: dict, details: dict) -> None:
    sample_dir = output_dir / row["model"] / Path(row["image"]).stem / str(
        row["rotation"]
    )
    original_rgb = details["original_crop_rgb"]
    corrected_rgb = details["corrected_crop_rgb"]
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
    imwrite(sample_dir / "input.png", original_bgr)
    imwrite(sample_dir / "corrected.png", corrected_bgr)

    visualization = corrected_bgr.copy()
    for index, box in enumerate(details["boxes"]):
        crop_points = np.asarray(box, dtype=np.float32).reshape(4, 2)
        draw_points = np.rint(crop_points).astype(np.int32)
        cv2.polylines(visualization, [draw_points], True, (0, 255, 0), 1)
        cv2.putText(
            visualization,
            str(index),
            tuple(draw_points[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        text_crop = get_rotate_crop_image(corrected_bgr, crop_points.copy())
        imwrite(sample_dir / f"text_crop_{index}.png", text_crop)
    imwrite(sample_dir / "det_boxes.png", visualization)
    debug_json = {
        key: value
        for key, value in details.items()
        if key not in {"original_crop_rgb", "corrected_crop_rgb"}
    }
    debug_json.update(row)
    (sample_dir / "result.json").write_text(
        json.dumps(debug_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_model(
    name: str,
    config_path: Path,
    registry_path: Path,
    samples: list,
    output_dir: Path,
):
    ocr = load_ocr(config_path, registry_path)
    rows = []
    try:
        for sample in samples:
            bgr = imread(sample["image"])
            upright_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for rotation in ANGLES:
                rgb = rotate_input(upright_rgb, rotation)
                started = time.perf_counter()
                details = ocr._read_details(
                    rgb,
                    (0, 0, rgb.shape[1], rgb.shape[0]),
                    include_images=True,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                predicted_class, match_score = (
                    ocr._classify(details["text"])
                    if details["text"]
                    else (None, 0.0)
                )
                row = {
                    "model": name,
                    "orientation_enabled": ocr._doc_orientation is not None,
                    "image": sample["image"].name,
                    "gt_text": sample["gt_text"],
                    "gt_class": sample["gt_class"],
                    "rotation": rotation,
                    "orientation_prediction": details["orientation_angle"],
                    "orientation_confidence": round(
                        details["orientation_confidence"], 6
                    ),
                    "orientation_applied": details["orientation_applied"],
                    "detected_box_count": len(details["boxes"]),
                    "detected_boxes": json.dumps(details["boxes"]),
                    "recognitions": json.dumps(
                        details["recognitions"], ensure_ascii=False
                    ),
                    "final_text": details["text"],
                    "ocr_confidence": round(details["ocr_confidence"], 6),
                    "matched_class": predicted_class or "",
                    "match_score": round(float(match_score), 6),
                    "exact_text": details["text"] == sample["gt_text"],
                    "character_recall": round(
                        lcs_recall(sample["gt_text"], details["text"]), 6
                    ),
                    "class_success": predicted_class == sample["gt_class"],
                    "elapsed_ms": round(elapsed_ms, 3),
                }
                rows.append(row)
                save_debug(output_dir, row, details)
    finally:
        ocr.close()
    return rows


def summarize(rows: list) -> dict:
    result = {}
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        by_rotation = {}
        for angle in ANGLES:
            angle_rows = [row for row in selected if row["rotation"] == angle]
            by_rotation[str(angle)] = {
                "total": len(angle_rows),
                "exact_text": sum(row["exact_text"] for row in angle_rows),
                "class_success": sum(
                    row["class_success"] for row in angle_rows
                ),
                "mean_character_recall": round(
                    sum(row["character_recall"] for row in angle_rows)
                    / len(angle_rows),
                    6,
                ),
            }
        model_summary = {
            "total": len(selected),
            "exact_text": sum(row["exact_text"] for row in selected),
            "class_success": sum(row["class_success"] for row in selected),
            "empty_text": sum(not row["final_text"] for row in selected),
            "single_character_text": sum(
                len(row["final_text"]) == 1 for row in selected
            ),
            "mean_character_recall": round(
                sum(row["character_recall"] for row in selected) / len(selected),
                6,
            ),
            "mean_elapsed_ms": round(
                sum(row["elapsed_ms"] for row in selected) / len(selected), 3
            ),
            "by_rotation": by_rotation,
        }
        if selected[0]["orientation_enabled"]:
            model_summary["orientation_accuracy"] = sum(
                row["orientation_prediction"] == row["rotation"]
                for row in selected
            )
            model_summary["orientation_correct_and_applied"] = sum(
                row["orientation_applied"]
                and row["orientation_prediction"] == row["rotation"]
                for row in selected
            )
            model_summary["high_confidence_orientation_errors"] = sum(
                row["orientation_confidence"] >= 0.8
                and row["orientation_prediction"] != row["rotation"]
                for row in selected
            )
            model_summary["orientation_confusion_matrix"] = {
                str(actual): {
                    str(predicted): sum(
                        row["rotation"] == actual
                        and row["orientation_prediction"] == predicted
                        for row in selected
                    )
                    for predicted in ANGLES
                }
                for actual in ANGLES
            }
            for angle in ANGLES:
                angle_rows = [
                    row for row in selected if row["rotation"] == angle
                ]
                by_rotation[str(angle)]["orientation_correct"] = sum(
                    row["orientation_prediction"] == angle
                    for row in angle_rows
                )
        result[model] = model_summary
    return result


def main() -> int:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    manifest_path = (
        resolve(args.manifest) if args.manifest else input_dir / "manifest.csv"
    )
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_manifest(input_dir, manifest_path)
    configs = [("ppocrv6", resolve(args.config))]
    if args.v6_no_orientation_config:
        configs.insert(
            0,
            (
                "ppocrv6_no_orientation",
                resolve(args.v6_no_orientation_config),
            ),
        )
    if args.baseline_config:
        configs.insert(0, ("ppocrv5", resolve(args.baseline_config)))

    rows = []
    registry_path = resolve(args.class_registry)
    for name, config_path in configs:
        rows.extend(
            run_model(name, config_path, registry_path, samples, output_dir)
        )
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
