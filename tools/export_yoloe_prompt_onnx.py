#!/usr/bin/env python3
"""Export a fixed-prompt YOLOE ONNX model.

This script is intended for deployment: text prompts are embedded once on the
development machine, injected into YOLOE with set_classes(), and then exported
as a fixed-class ONNX model. The resulting .onnx can be converted to .om with
the same filename stem for backend=auto runtime selection.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-prompt YOLOE ONNX weights.")
    parser.add_argument(
        "--source-weights",
        default="models/yoloe-26s-seg.pt",
        help="Source Ultralytics YOLOE .pt weights.",
    )
    parser.add_argument(
        "--output",
        default="models/yoloe-26s-open-set.onnx",
        help="Output ONNX path. The .onnx suffix is added if omitted.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Prompt labels to bake into the exported model, e.g. --labels book.",
    )
    parser.add_argument(
        "--prompt-embeddings",
        default=None,
        help="Optional .pt cache with {'labels', 'embeddings'} from precompute script.",
    )
    parser.add_argument(
        "--save-prompt-embeddings",
        default=None,
        help="Optional path to save generated embeddings for reuse.",
    )
    parser.add_argument(
        "--text-model",
        default="models/mobileclip2_b.ts",
        help="MobileCLIP2 TorchScript text encoder used when embeddings are not provided.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Static export image size.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset.")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu, mps, cuda:0.")
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Export with Ultralytics NMS included. Leave off if ATC cannot convert NMS.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export dynamic shape ONNX. Default is static shape for easier OM conversion.",
    )
    return parser.parse_args()


def ensure_text_model_available(text_model: Path) -> None:
    """Make Ultralytics resolve the local MobileCLIP2 file instead of downloading."""
    if not text_model.exists():
        raise FileNotFoundError(f"text model does not exist: {text_model}")

    from ultralytics.utils import SETTINGS

    SETTINGS["weights_dir"] = str(text_model.parent)
    target = Path(SETTINGS["weights_dir"]) / text_model.name
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(text_model, target)


def load_prompt_embeddings(path: Path, labels: Sequence[str]):
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        cached_labels = [str(label) for label in payload.get("labels", [])]
        if cached_labels and cached_labels != list(labels):
            raise ValueError(f"embedding labels mismatch: cache={cached_labels}, labels={list(labels)}")
        embeddings = payload.get("embeddings")
    else:
        embeddings = payload
    if embeddings is None:
        raise ValueError(f"prompt embedding file has no embeddings: {path}")
    return embeddings


def save_prompt_embeddings(path: Path, labels: Sequence[str], embeddings, source_weights: Path, text_model: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "labels": list(labels),
            "embeddings": embeddings.detach().cpu(),
            "weights": str(source_weights),
            "text_model": str(text_model),
        },
        path,
    )


def export_yoloe(args: argparse.Namespace) -> Path:
    from ultralytics import YOLOE

    source_weights = Path(args.source_weights).expanduser()
    if not source_weights.is_file():
        raise FileNotFoundError(f"source weights do not exist: {source_weights}")
    output = Path(args.output).expanduser()
    if output.suffix != ".onnx":
        output = output.with_suffix(".onnx")
    labels = [str(label) for label in args.labels]
    if not labels:
        raise ValueError("at least one label is required")

    text_model = Path(args.text_model).expanduser()
    model = YOLOE(str(source_weights))
    if args.device:
        model.to(args.device)

    if args.prompt_embeddings:
        embeddings = load_prompt_embeddings(Path(args.prompt_embeddings).expanduser(), labels)
    else:
        ensure_text_model_available(text_model)
        embeddings = model.get_text_pe(labels).detach().cpu()

    if args.save_prompt_embeddings:
        save_prompt_embeddings(
            Path(args.save_prompt_embeddings).expanduser(),
            labels,
            embeddings,
            source_weights,
            text_model,
        )

    model.set_classes(labels, embeddings=embeddings)
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            nms=bool(args.nms),
            dynamic=bool(args.dynamic),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output.resolve():
        shutil.move(str(exported), str(output))
    print(f"exported fixed-prompt YOLOE ONNX: {output}")
    print(f"labels={labels}")
    print(f"nms={bool(args.nms)} dynamic={bool(args.dynamic)} imgsz={args.imgsz} opset={args.opset}")
    return output


def main() -> int:
    export_yoloe(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
