from __future__ import annotations

import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

from .file_store import IMAGE_SUFFIXES, copy_input_pair, ensure_runtime_dirs, write_run_summary
from .stereo_api import StereoApiClient, StereoApiError


@dataclass
class TaskResult:
    success: bool
    message: str
    metadata: dict | None = None


class PipelineRunner:
    def __init__(self, config: dict, logger=None, status_callback=None, progress_callback=None, stop_flag=None):
        self.config = config
        self.logger = logger or (lambda message: None)
        self.status_callback = status_callback or (lambda *args: None)
        self.progress_callback = progress_callback or (lambda current, total: None)
        self.stop_flag = stop_flag or (lambda: False)

    def run(self) -> TaskResult:
        try:
            return self._run_once()
        except Exception as exc:  # noqa: BLE001
            self.logger(f"处理失败：{exc}")
            self.logger(traceback.format_exc())
            return TaskResult(False, str(exc))

    @staticmethod
    def _build_camera_query_candidates(sample_name: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                candidates.append(text)

        add(sample_name)
        if "_" in sample_name:
            add(sample_name.split("_")[0])

        match = re.match(r"([A-Za-z][A-Za-z0-9]*\d+)", sample_name)
        if match:
            add(match.group(1))

        trimmed = re.sub(r"_\d{8}(?:_\d{4,6})?(?:_\d+)?$", "", sample_name)
        if trimmed != sample_name:
            add(trimmed)
            if "_" in trimmed:
                add(trimmed.split("_")[0])

        return candidates

    def _collect_raw_images(self, input_root: Path) -> list[Path]:
        selected_files = self.config.get("selected_files") or []
        if selected_files:
            return [
                Path(file_path)
                for file_path in selected_files
                if Path(file_path).is_file() and Path(file_path).suffix.lower() in IMAGE_SUFFIXES
            ]
        return sorted(path for path in input_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

    def _run_once(self) -> TaskResult:
        model_key = str(self.config.get("model_selection", "fs")).strip().lower()
        input_dir = self.config["paths"]["input_dir"]
        output_dir = self.config["paths"]["output_dir"]
        run_name = (
            f"{'FoundationStereo' if model_key == 'fs' else 'S2M2StereoMatching'}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        input_root, output_layout = ensure_runtime_dirs(input_dir, output_dir, run_name)
        preprocess_url = self.config["servers"]["preprocess"]
        raw_images = self._collect_raw_images(input_root)
        if not raw_images:
            raise FileNotFoundError(f"输入目录中没有可处理的图像：{input_root}")

        preprocess_client = StereoApiClient(preprocess_url)
        server_url = self.config["servers"][model_key]
        stereo_client = StereoApiClient(server_url)
        params = self.config["s2m2_params"] if model_key == "s2m2" else self.config["fs_params"]
        total = len(raw_images)
        self.progress_callback(0, total)

        try:
            status_text, timer_text = stereo_client.get_status(model_key)
            self.logger(f"深度服务状态：{status_text} | {timer_text}")
        except Exception as exc:  # noqa: BLE001
            self.logger(f"深度服务状态检查失败，继续尝试处理：{exc}")

        success_samples: list[str] = []
        failure_messages: list[str] = []
        failure_samples: list[dict] = []
        last_metadata = None
        cache_dir = output_layout.cache_dir

        for index, image_path in enumerate(raw_images, start=1):
            sample_name = image_path.stem or datetime.now().strftime("sample_%Y%m%d_%H%M%S")

            if self.stop_flag():
                cancel_message = f"任务已取消，已处理 {index - 1}/{total} 个样本。"
                summary_path = self._write_run_summary(
                    output_layout=output_layout,
                    run_name=run_name,
                    input_root=input_root,
                    total=total,
                    model_key=model_key,
                    server_url=server_url,
                    preprocess_url=preprocess_url,
                    raw_images=raw_images,
                    success_samples=success_samples,
                    failure_samples=failure_samples,
                    stopped=True,
                )
                return TaskResult(
                    False,
                    cancel_message,
                    {
                        "run_name": run_name,
                        "run_dir": str(output_layout.run_output),
                        "summary_path": str(summary_path),
                        "failure_count": len(failure_samples),
                        "failed_samples": failure_samples,
                    },
                )

            try:
                self.logger(f"[{index}/{total}] 开始处理样本：{sample_name}")
                self.status_callback("connecting", sample_name)

                query_candidates = self._build_camera_query_candidates(sample_name)
                params_text = ""
                last_query_error = ""
                for query_name in query_candidates:
                    try:
                        self.logger(f"[{sample_name}] 尝试匹配相机参数 Key: {query_name}")
                        params_text = preprocess_client.query_camera_params(query_name)
                        self.logger(f"[{sample_name}] 相机参数匹配成功 Key: {query_name}")
                        break
                    except StereoApiError as exc:
                        last_query_error = str(exc)

                if not params_text:
                    tried = "、".join(query_candidates)
                    raise StereoApiError(
                        f"未找到相机参数。已尝试 Key: {tried}"
                        + (f" | 最后一次返回：{last_query_error}" if last_query_error else "")
                    )

                left_image, right_image, intrinsic_text = preprocess_client.preprocess_image(
                    image_path,
                    params_text,
                    cache_dir,
                    sample_name,
                )
                copied_inputs = copy_input_pair(
                    cache_dir,
                    sample_name,
                    left_image,
                    right_image,
                    source_image=image_path,
                )

                self.status_callback("running", sample_name)
                start_time = perf_counter()
                remote_outputs = stereo_client.run_prediction(model_key, left_image, right_image, intrinsic_text, params)

                self.logger(f"[{sample_name}] 深度模型完成，开始保存结果")
                saved_outputs = {
                    "disp_vis": stereo_client.save_remote_output(
                        remote_outputs.get("disp_vis"),
                        output_layout.disparity_vis_dir / f"{sample_name}_disp_vis.png",
                    ),
                    "disp_npy": stereo_client.save_remote_output(
                        remote_outputs.get("disp_npy"),
                        output_layout.disparity_npy_dir / f"{sample_name}_disp.npy",
                    ),
                    "depth_npy": stereo_client.save_remote_output(
                        remote_outputs.get("depth_npy"),
                        output_layout.depth_npy_dir / f"{sample_name}_depth.npy",
                    ),
                }

                pointcloud_path = remote_outputs.get("pointcloud")
                saved_outputs["pointcloud"] = (
                    stereo_client.save_remote_output(
                        pointcloud_path,
                        output_layout.pointcloud_dir / f"{sample_name}.ply",
                    )
                    if pointcloud_path
                    else ""
                )

                elapsed = round(perf_counter() - start_time, 2)
                metadata = {
                    "sample_name": sample_name,
                    "model": model_key,
                    "run_name": run_name,
                    "run_dir": str(output_layout.run_output),
                    "server_url": server_url,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_seconds": elapsed,
                    "inputs": copied_inputs,
                    "intrinsic_text": intrinsic_text,
                    "parameters": params,
                    "outputs": saved_outputs,
                    "metadata_path": "",
                }
                last_metadata = metadata
                success_samples.append(sample_name)
                self.status_callback("finished", sample_name)
                self.logger(f"[{sample_name}] 处理完成，耗时 {elapsed} 秒")
                self.logger(f"[{sample_name}] 输出目录：{output_layout.run_output}")
            except Exception as exc:  # noqa: BLE001
                failure_samples.append(
                    {
                        "sample_name": sample_name,
                        "source_image": str(image_path),
                        "error": str(exc),
                    }
                )
                failure_messages.append(f"{sample_name}: {exc}")
                self.logger(f"[{sample_name}] 处理失败：{exc}")

            self.progress_callback(index, total)

        summary_path = self._write_run_summary(
            output_layout=output_layout,
            run_name=run_name,
            input_root=input_root,
            total=total,
            model_key=model_key,
            server_url=server_url,
            preprocess_url=preprocess_url,
            raw_images=raw_images,
            success_samples=success_samples,
            failure_samples=failure_samples,
            stopped=False,
        )

        if not success_samples:
            detail = failure_messages[-1] if failure_messages else "未生成任何结果。"
            return TaskResult(
                False,
                f"批处理失败：{detail}",
                {
                    "run_name": run_name,
                    "run_dir": str(output_layout.run_output),
                    "summary_path": str(summary_path),
                    "failure_count": len(failure_samples),
                    "failed_samples": failure_samples,
                },
            )

        summary = f"处理完成：成功 {len(success_samples)}/{total}"
        if failure_messages:
            summary += f"，失败 {len(failure_messages)}"
            self.logger("失败样本：" + "；".join(failure_messages))
        self.logger(f"本轮输出目录：{output_layout.run_output}")

        result_metadata = dict(last_metadata or {})
        result_metadata.update(
            {
                "run_name": run_name,
                "run_dir": str(output_layout.run_output),
                "summary_path": str(summary_path),
                "failure_count": len(failure_samples),
                "failed_samples": failure_samples,
            }
        )
        return TaskResult(True, summary, result_metadata)

    @staticmethod
    def _write_run_summary(
        *,
        output_layout,
        run_name: str,
        input_root: Path,
        total: int,
        model_key: str,
        server_url: str,
        preprocess_url: str,
        raw_images: list[Path],
        success_samples: list[str],
        failure_samples: list[dict],
        stopped: bool,
    ) -> Path:
        run_summary = {
            "run_name": run_name,
            "run_dir": str(output_layout.run_output),
            "input_dir": str(input_root),
            "output_dir": str(output_layout.base_output),
            "model": model_key,
            "server_url": server_url,
            "preprocess_url": preprocess_url,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "total_samples": total,
            "success_count": len(success_samples),
            "failure_count": len(failure_samples),
            "stopped": stopped,
            "success_samples": success_samples,
            "failed_samples": failure_samples,
            "selected_files": [str(path) for path in raw_images],
            "retryable_failed_files": [
                item["source_image"]
                for item in failure_samples
                if str(item.get("source_image", "")).strip()
            ],
        }
        return write_run_summary(output_layout.run_output, run_summary)
