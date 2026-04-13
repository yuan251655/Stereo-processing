from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

from .file_store import copy_input_pair, ensure_runtime_dirs, write_metadata
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

    def _run_once(self) -> TaskResult:
        model_key = str(self.config.get("model_selection", "fs")).strip().lower()
        session = self.config.get("session", {})
        left_image = Path(str(session.get("left_image", "")).strip())
        right_image = Path(str(session.get("right_image", "")).strip())
        sample_name = str(session.get("sample_name", "")).strip() or datetime.now().strftime("sample_%Y%m%d_%H%M%S")
        intrinsic_text = str(session.get("intrinsic_text", "")).strip()

        if not left_image.exists():
            raise FileNotFoundError(f"左图不存在：{left_image}")
        if not right_image.exists():
            raise FileNotFoundError(f"右图不存在：{right_image}")
        if not intrinsic_text:
            raise ValueError("内参与基线不能为空。")

        input_dir = self.config["paths"]["input_dir"]
        output_dir = self.config["paths"]["output_dir"]
        input_root, output_layout = ensure_runtime_dirs(input_dir, output_dir)
        self.progress_callback(0, 1)

        if session.get("save_input_copy", True):
            self.logger("保存原始双目输入到 input 目录...")
            copied_inputs = copy_input_pair(input_root, sample_name, left_image, right_image)
        else:
            copied_inputs = {
                "sample_name": sample_name,
                "left_image": str(left_image),
                "right_image": str(right_image),
            }

        if self.stop_flag():
            return TaskResult(False, "任务已取消。")

        self.status_callback("connecting", sample_name)
        server_url = self.config["servers"][model_key]
        client = StereoApiClient(server_url)

        try:
            status_text, timer_text = client.get_status(model_key)
            self.logger(f"服务状态：{status_text} | {timer_text}")
        except Exception as exc:  # noqa: BLE001
            self.logger(f"状态检查失败，继续尝试推理：{exc}")

        params = self.config["s2m2_params"] if model_key == "s2m2" else self.config["fs_params"]
        self.logger(f"开始调用 {model_key.upper()} 模型...")
        self.status_callback("running", sample_name)
        start_time = perf_counter()

        try:
            remote_outputs = client.run_prediction(model_key, left_image, right_image, intrinsic_text, params)
        except StereoApiError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StereoApiError(f"模型调用失败：{exc}") from exc

        if self.stop_flag():
            return TaskResult(False, "任务已取消。")

        self.logger("模型处理完成，开始保存结果文件...")
        saved_outputs = {
            "disp_vis": client.save_remote_output(
                remote_outputs.get("disp_vis"),
                output_layout.disparity_vis_dir / f"{sample_name}_disp_vis.png",
            ),
            "disp_png": client.save_remote_output(
                remote_outputs.get("disp_png"),
                output_layout.disparity_png_dir / f"{sample_name}_disp.png",
            ),
            "disp_npy": client.save_remote_output(
                remote_outputs.get("disp_npy"),
                output_layout.disparity_npy_dir / f"{sample_name}_disp.npy",
            ),
            "depth_npy": client.save_remote_output(
                remote_outputs.get("depth_npy"),
                output_layout.depth_npy_dir / f"{sample_name}_depth.npy",
            ),
        }

        pointcloud_path = remote_outputs.get("pointcloud")
        if pointcloud_path:
            saved_outputs["pointcloud"] = client.save_remote_output(
                pointcloud_path,
                output_layout.pointcloud_dir / f"{sample_name}.ply",
            )
        else:
            saved_outputs["pointcloud"] = ""

        conf_vis = remote_outputs.get("conf_vis")
        conf_npy = remote_outputs.get("conf_npy")
        if conf_vis:
            saved_outputs["conf_vis"] = client.save_remote_output(
                conf_vis,
                output_layout.confidence_vis_dir / f"{sample_name}_conf_vis.png",
            )
        else:
            saved_outputs["conf_vis"] = ""

        if conf_npy:
            saved_outputs["conf_npy"] = client.save_remote_output(
                conf_npy,
                output_layout.confidence_npy_dir / f"{sample_name}_conf.npy",
            )
        else:
            saved_outputs["conf_npy"] = ""

        elapsed = round(perf_counter() - start_time, 2)
        metadata = {
            "sample_name": sample_name,
            "model": model_key,
            "server_url": server_url,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed,
            "inputs": copied_inputs,
            "intrinsic_text": intrinsic_text,
            "parameters": params,
            "outputs": saved_outputs,
        }
        metadata_path = write_metadata(output_layout.metadata_dir, sample_name, metadata)
        metadata["metadata_path"] = str(metadata_path)

        self.progress_callback(1, 1)
        self.status_callback("finished", sample_name)
        self.logger(f"处理完成：{sample_name}，总耗时 {elapsed} 秒。")
        self.logger(f"结果清单已写入：{metadata_path}")
        return TaskResult(True, "处理完成", metadata)

