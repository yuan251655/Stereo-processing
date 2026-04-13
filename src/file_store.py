from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class OutputLayout:
    base_output: Path
    disparity_vis_dir: Path
    disparity_png_dir: Path
    disparity_npy_dir: Path
    depth_npy_dir: Path
    pointcloud_dir: Path
    confidence_vis_dir: Path
    confidence_npy_dir: Path
    metadata_dir: Path


def ensure_runtime_dirs(input_dir: str | Path, output_dir: str | Path) -> tuple[Path, OutputLayout]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    for path in [
        input_path,
        input_path / "left",
        input_path / "right",
        input_path / "pairs",
        output_path,
        output_path / "disparity_vis",
        output_path / "disparity_png",
        output_path / "disparity_npy",
        output_path / "depth_npy",
        output_path / "pointcloud",
        output_path / "confidence_vis",
        output_path / "confidence_npy",
        output_path / "metadata",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    layout = OutputLayout(
        base_output=output_path,
        disparity_vis_dir=output_path / "disparity_vis",
        disparity_png_dir=output_path / "disparity_png",
        disparity_npy_dir=output_path / "disparity_npy",
        depth_npy_dir=output_path / "depth_npy",
        pointcloud_dir=output_path / "pointcloud",
        confidence_vis_dir=output_path / "confidence_vis",
        confidence_npy_dir=output_path / "confidence_npy",
        metadata_dir=output_path / "metadata",
    )
    return input_path, layout


def copy_input_pair(input_dir: str | Path, sample_name: str, left_image: str | Path, right_image: str | Path) -> dict:
    input_path = Path(input_dir)
    left_src = Path(left_image)
    right_src = Path(right_image)

    left_target = input_path / "left" / f"{sample_name}{left_src.suffix.lower()}"
    right_target = input_path / "right" / f"{sample_name}{right_src.suffix.lower()}"
    pair_manifest = input_path / "pairs" / f"{sample_name}.json"

    shutil.copy2(left_src, left_target)
    shutil.copy2(right_src, right_target)

    payload = {
        "sample_name": sample_name,
        "left_image": str(left_target),
        "right_image": str(right_target),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    pair_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_metadata(metadata_dir: Path, sample_name: str, payload: dict) -> Path:
    metadata_path = metadata_dir / f"{sample_name}.json"
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def list_saved_results(output_dir: str | Path) -> list[dict]:
    metadata_dir = Path(output_dir) / "metadata"
    if not metadata_dir.exists():
        return []

    records = []
    for json_path in sorted(metadata_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["metadata_path"] = str(json_path)
        records.append(payload)
    return records

