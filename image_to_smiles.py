#!/usr/bin/env python3
"""
Convert chemical structure images to SMILES using MolScribe + RDKit.

Examples:
  python image_to_smiles.py --input Picture.png
  python image_to_smiles.py --input Picture.png --canonical --output smiles.csv
  python image_to_smiles.py --input ./images --output batch_smiles.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import torch
from molscribe import MolScribe
from rdkit import Chem

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CKPT = "ckpts/swin_base_char_aux_1m680k.pth"

# Silence known non-critical torch warnings in inference mode.
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image to SMILES converter (MolScribe)")
    parser.add_argument("--input", required=True, help="Input image path OR directory")
    parser.add_argument("--output", default="", help="Output CSV path (optional)")
    parser.add_argument("--md-output", default="", help="Output Markdown report path (optional)")
    parser.add_argument("--model-path", default=DEFAULT_CKPT, help=f"MolScribe checkpoint path (default: {DEFAULT_CKPT})")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable image preprocessing")
    parser.add_argument("--canonical", action="store_true", help="Output canonical SMILES")
    parser.add_argument("--keep-hydrogens", action="store_true", help="Keep explicit hydrogens")
    parser.add_argument(
        "--keep-disconnected",
        action="store_true",
        help="Keep disconnected fragments in SMILES (default keeps only largest fragment)",
    )
    parser.add_argument("--with-confidence", action="store_true", help="Include model confidence")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--no-segment", action="store_true", help="Disable molecule segmentation (run OCSR on full image)")
    parser.add_argument("--segment-padding", type=int, default=12, help="Padding around each segmented molecule box")
    parser.add_argument("--segment-min-area-ratio", type=float, default=0.01, help="Minimum connected-area ratio for molecule segments")
    parser.add_argument(
        "--manual-ids",
        default="",
        help="Comma-separated IDs for segmented molecules in visual order, e.g. 1,2,3,4",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed errors")
    return parser.parse_args()


def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in ALLOWED_EXTS:
            raise ValueError(f"Unsupported file extension: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    images = [
        p
        for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    ]
    if not images:
        raise FileNotFoundError(f"No images found under: {input_path}")
    return images


def preprocess_image(src: Path) -> Path:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {src}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)
    bw = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )

    white_ratio = float((bw == 255).sum()) / bw.size
    if white_ratio < 0.5:
        bw = cv2.bitwise_not(bw)

    bw = cv2.copyMakeBorder(bw, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)

    tmp = tempfile.NamedTemporaryFile(prefix="ocsr_", suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    if not cv2.imwrite(str(tmp_path), bw):
        raise RuntimeError(f"Failed to write temporary image: {tmp_path}")
    return tmp_path


def normalize_smiles(smiles: str, canonical: bool, keep_h: bool, keep_disconnected: bool) -> Tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", "RDKit failed to parse predicted SMILES"

    if not keep_disconnected:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if frags:
            # One segmented crop should map to one molecule; keep the largest component
            # to drop OCR artifacts such as stray ions or tiny text-derived fragments.
            mol = max(
                frags,
                key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()),
            )

    if not keep_h:
        mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, canonical=canonical), "ok"


def segment_molecule_crops(
    image_path: Path,
    padding: int = 12,
    min_area_ratio: float = 0.01,
) -> List[Tuple[str, Path, Tuple[int, int, int, int]]]:
    """
    Segment one image into per-molecule crops.
    Returns list of (segment_id, temporary_crop_path, bbox[x0,y0,x1,y1]).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image for segmentation: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        5,
    )

    # Merge strokes that belong to the same molecular drawing.
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 21))
    merged = cv2.dilate(bw, merge_kernel, iterations=1)

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(merged, 8)
    min_area = max(2000, int(h * w * max(0.001, min_area_ratio)))
    boxes: List[Tuple[int, int, int, int, int]] = []

    for i in range(1, n_labels):
        x, y, bw_box, bh_box, area = stats[i]
        if area < min_area or bw_box < 60 or bh_box < 60:
            continue
        boxes.append((x, y, bw_box, bh_box, area))

    if not boxes:
        boxes = [(0, 0, w, h, w * h)]

    boxes.sort(key=lambda b: (b[1], b[0]))
    crops: List[Tuple[str, Path, Tuple[int, int, int, int]]] = []
    for idx, (x, y, bw_box, bh_box, _area) in enumerate(boxes, start=1):
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw_box + padding)
        y1 = min(h, y + bh_box + padding)
        crop = img[y0:y1, x0:x1]

        tmp = tempfile.NamedTemporaryFile(prefix=f"mol_{idx}_", suffix=".png", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        if not cv2.imwrite(str(tmp_path), crop):
            raise RuntimeError(f"Failed to write segmented crop: {tmp_path}")
        crops.append((f"mol{idx}", tmp_path, (x0, y0, x1, y1)))

    return crops


def assign_ids_to_segments(
    segments: List[Tuple[str, Path, Tuple[int, int, int, int]]],
    manual_ids: List[str],
) -> List[Tuple[str, Path]]:
    """Assign display IDs to segments, preferring manual IDs, then fallback."""
    if manual_ids:
        assigned: List[Tuple[str, Path]] = []
        for i, (_seg_id, seg_path, _bbox) in enumerate(segments):
            label = manual_ids[i] if i < len(manual_ids) else f"mol{i+1}"
            assigned.append((label, seg_path))
        return assigned

    return [(f"mol{i+1}", seg_path) for i, (_seg_id, seg_path, _bbox) in enumerate(segments)]


def image_to_smiles(
    predictor: MolScribe,
    image_path: Path,
    preprocess: bool,
    canonical: bool,
    keep_h: bool,
    keep_disconnected: bool,
    with_confidence: bool,
) -> Tuple[str, str, str, str]:
    """Return (smiles, raw_smiles, confidence, status)."""
    tmp_path: Path | None = None
    best_raw = ""
    best_conf = ""
    best_status = "MolScribe returned empty SMILES"
    tried = [preprocess] if not preprocess else [True, False]

    for use_preprocess in tried:
        try:
            work_path = image_path
            if use_preprocess:
                tmp_path = preprocess_image(image_path)
                work_path = tmp_path

            pred = predictor.predict_image_file(
                str(work_path),
                return_confidence=with_confidence,
            )
            raw_smiles = (pred.get("smiles") or "").strip()
            conf_str = str(pred.get("confidence", ""))

            if raw_smiles:
                best_raw = raw_smiles
                best_conf = conf_str
                best_status = "raw_smiles_unparsed"

                smiles, status = normalize_smiles(
                    raw_smiles,
                    canonical=canonical,
                    keep_h=keep_h,
                    keep_disconnected=keep_disconnected,
                )
                if status == "ok":
                    return smiles, raw_smiles, conf_str, "ok"
                best_status = status
            else:
                best_status = "MolScribe returned empty SMILES"
        except Exception as exc:
            best_status = str(exc)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                tmp_path = None

    if best_raw:
        return best_raw, best_raw, best_conf, "raw_smiles_unparsed"
    return "", "", best_conf, best_status


def write_csv(rows: Iterable[Tuple[str, str, str, str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "smiles", "raw_smiles", "confidence", "status"])
        writer.writerows(rows)


def write_markdown(rows: List[Tuple[str, str, str, str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok_count = sum(1 for _, _, _, _, st in rows if st in {"ok", "raw_smiles_unparsed"})
    fail_count = len(rows) - ok_count

    lines: List[str] = []
    lines.append("# 化学式识别结果")
    lines.append("")
    lines.append(f"- 总数: {len(rows)}")
    lines.append(f"- 成功: {ok_count}")
    lines.append(f"- 失败: {fail_count}")
    lines.append("")
    lines.append("## 一行一个对应关系（化学式编号 -> SMILES）")
    lines.append("")
    for idx, (image, smiles, raw_smiles, _confidence, _status) in enumerate(rows, start=1):
        image_name = Path(image).name
        shown_id = image.split("#", 1)[1] if "#" in image else str(idx)
        value = smiles if smiles else (raw_smiles if raw_smiles else "-")
        lines.append(f"- 化学式{shown_id}（{image_name}） -> `{value}`")

    lines.append("")
    lines.append("## 详细信息")
    lines.append("")
    for idx, (image, smiles, raw_smiles, confidence, status) in enumerate(rows, start=1):
        image_name = Path(image).name
        shown_id = image.split("#", 1)[1] if "#" in image else str(idx)
        lines.append(f"### 化学式{shown_id}：{image_name}")
        lines.append(f"- 图片路径: `{image}`")
        lines.append(f"- 状态: `{status}`")
        lines.append(f"- 置信度: `{confidence if confidence else '-'}`")
        lines.append("- 标准化 SMILES:")
        lines.append("```text")
        lines.append(smiles if smiles else "-")
        lines.append("```")
        lines.append("- 原始预测 SMILES:")
        lines.append("```text")
        lines.append(raw_smiles if raw_smiles else "-")
        lines.append("```")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    if not model_path.exists():
        print(f"[ERROR] Model checkpoint not found: {model_path}", file=sys.stderr)
        print(
            "Hint: mkdir -p ckpts && wget -O ckpts/swin_base_char_aux_1m680k.pth "
            "https://hf-mirror.com/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth",
            file=sys.stderr,
        )
        return 2

    try:
        images = collect_images(input_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    predictor = MolScribe(str(model_path), device=device)

    manual_ids = [x.strip() for x in args.manual_ids.split(",") if x.strip()]
    rows: List[Tuple[str, str, str, str, str]] = []
    for img_path in images:
        segment_items: List[Tuple[str, Path, Tuple[int, int, int, int]]] = []
        try:
            if args.no_segment:
                segment_items = [("mol1", img_path, (0, 0, 0, 0))]
            else:
                segment_items = segment_molecule_crops(
                    img_path,
                    padding=args.segment_padding,
                    min_area_ratio=args.segment_min_area_ratio,
                )
                print(f"[SEG] {img_path.name}: detected {len(segment_items)} molecules")
        except Exception as exc:
            print(f"[FAIL] {img_path.name} segmentation: {exc}", file=sys.stderr)
            continue

        assigned_segments = assign_ids_to_segments(segment_items, manual_ids)

        for seg_id, seg_path in assigned_segments:
            smiles, raw_smiles, confidence, status = image_to_smiles(
                predictor=predictor,
                image_path=seg_path,
                preprocess=not args.no_preprocess,
                canonical=args.canonical,
                keep_h=args.keep_hydrogens,
                keep_disconnected=args.keep_disconnected,
                with_confidence=args.with_confidence,
            )
            source_label = f"{img_path.name}#{seg_id}"
            rows.append((source_label, smiles, raw_smiles, confidence, status))

            if status in {"ok", "raw_smiles_unparsed"}:
                extra = f" (confidence={confidence})" if confidence else ""
                tag = "OK" if status == "ok" else "RAW"
                print(f"[{tag}] {source_label}: {smiles}{extra}")
            elif args.verbose:
                print(f"[FAIL] {source_label}: {status}", file=sys.stderr)
            else:
                print(f"[FAIL] {source_label}", file=sys.stderr)

            if seg_path != img_path:
                try:
                    seg_path.unlink(missing_ok=True)
                except OSError:
                    pass

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        write_csv(rows, out_path)
        print(f"\nSaved CSV: {out_path}")
    else:
        print("\nimage,smiles,raw_smiles,confidence,status")
        for r in rows:
            print(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}")

    if args.md_output:
        md_path = Path(args.md_output).expanduser().resolve()
        write_markdown(rows, md_path)
        print(f"Saved Markdown: {md_path}")

    ok_count = sum(1 for _, _, _, _, st in rows if st in {"ok", "raw_smiles_unparsed"})
    print(f"\nDone. success={ok_count}/{len(rows)}")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
