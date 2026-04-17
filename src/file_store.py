from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class OutputLayout:
    base_output: Path
    run_output: Path
    cache_dir: Path
    disparity_vis_dir: Path
    disparity_npy_dir: Path
    depth_npy_dir: Path
    pointcloud_dir: Path


def ensure_runtime_dirs(input_dir: str | Path, output_dir: str | Path, run_name: str) -> tuple[Path, OutputLayout]:
    input_path = Path(input_dir)
    output_root = Path(output_dir)
    run_output = output_root / run_name
    cache_dir = Path(tempfile.gettempdir()) / "stereo_processing_cache" / run_name

    for path in [
        input_path,
        output_root,
        run_output,
        cache_dir,
        run_output / "disparity_vis",
        run_output / "disparity_npy",
        run_output / "depth_npy",
        run_output / "pointcloud",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    layout = OutputLayout(
        base_output=output_root,
        run_output=run_output,
        cache_dir=cache_dir,
        disparity_vis_dir=run_output / "disparity_vis",
        disparity_npy_dir=run_output / "disparity_npy",
        depth_npy_dir=run_output / "depth_npy",
        pointcloud_dir=run_output / "pointcloud",
    )
    return input_path, layout


def copy_input_pair(
    cache_dir: str | Path,
    sample_name: str,
    left_image: str | Path,
    right_image: str | Path,
    source_image: str | Path | None = None,
) -> dict:
    cache_path = Path(cache_dir)
    left_src = Path(left_image)
    right_src = Path(right_image)

    left_target = cache_path / f"{sample_name}_left{left_src.suffix.lower()}"
    right_target = cache_path / f"{sample_name}_right{right_src.suffix.lower()}"
    pair_manifest = cache_path / f"{sample_name}_pair.json"

    shutil.copy2(left_src, left_target)
    shutil.copy2(right_src, right_target)

    payload = {
        "sample_name": sample_name,
        "left_image": str(left_target),
        "right_image": str(right_target),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if source_image:
        payload["source_image"] = str(source_image)
    pair_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_metadata(metadata_dir: Path, sample_name: str, payload: dict) -> Path:
    metadata_path = metadata_dir / f"{sample_name}.json"
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def write_run_summary(run_dir: Path, payload: dict) -> Path:
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def load_latest_run_summary(output_dir: str | Path) -> dict | None:
    output_root = Path(output_dir)
    if not output_root.exists():
        return None

    run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
    for run_dir in sorted(run_dirs, key=lambda item: item.stat().st_mtime, reverse=True):
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return None


def list_saved_results(output_dir: str | Path) -> list[dict]:
    output_root = Path(output_dir)
    if not output_root.exists():
        return []

    records = []
    for run_dir in sorted((path for path in output_root.iterdir() if path.is_dir()), key=lambda item: item.stat().st_mtime, reverse=True):
        disparity_vis_dir = run_dir / "disparity_vis"
        disparity_npy_dir = run_dir / "disparity_npy"
        depth_npy_dir = run_dir / "depth_npy"
        pointcloud_dir = run_dir / "pointcloud"

        if not any(path.exists() for path in [disparity_vis_dir, disparity_npy_dir, depth_npy_dir, pointcloud_dir]):
            continue

        sample_map: dict[str, dict] = {}

        def ensure_record(sample_name: str) -> dict:
            if sample_name not in sample_map:
                sample_map[sample_name] = {
                    "sample_name": sample_name,
                    "model": "s2m2" if run_dir.name.lower().startswith("s2m2") else "fs",
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "created_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
                    "elapsed_seconds": None,
                    "server_url": "",
                    "parameters": {},
                    "inputs": {},
                    "outputs": {
                        "disp_vis": "",
                        "disp_npy": "",
                        "depth_npy": "",
                        "pointcloud": "",
                    },
                    "metadata_path": "",
                }
            return sample_map[sample_name]

        if disparity_vis_dir.exists():
            for path in disparity_vis_dir.glob("*_disp_vis.png"):
                sample_name = path.stem.removesuffix("_disp_vis")
                ensure_record(sample_name)["outputs"]["disp_vis"] = str(path)

        if disparity_npy_dir.exists():
            for path in disparity_npy_dir.glob("*_disp.npy"):
                sample_name = path.stem.removesuffix("_disp")
                ensure_record(sample_name)["outputs"]["disp_npy"] = str(path)

        if depth_npy_dir.exists():
            for path in depth_npy_dir.glob("*_depth.npy"):
                sample_name = path.stem.removesuffix("_depth")
                ensure_record(sample_name)["outputs"]["depth_npy"] = str(path)

        if pointcloud_dir.exists():
            for path in pointcloud_dir.glob("*.ply"):
                sample_name = path.stem
                ensure_record(sample_name)["outputs"]["pointcloud"] = str(path)

        records.extend(sample_map.values())

    records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return records
