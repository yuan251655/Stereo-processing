from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import requests
from gradio_client import Client


class StereoApiError(RuntimeError):
    pass


class StereoApiClient:
    def __init__(self, server_url: str, timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def get_status(self, model_key: str) -> tuple[str, str]:
        api_name = "/get_status" if model_key == "s2m2" else "/get_system_status"
        client = Client(self.server_url)
        result = client.predict(api_name=api_name)
        if isinstance(result, tuple) and len(result) >= 2:
            return str(result[0]), str(result[1])
        raise StereoApiError("状态接口返回格式异常。")

    def upload_file(self, file_path: str | Path) -> str:
        upload_url = f"{self.server_url}/gradio_api/upload"
        with Path(file_path).open("rb") as file:
            response = requests.post(
                upload_url,
                files={"files": (Path(file_path).name, file)},
                timeout=self.timeout,
            )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            raise StereoApiError(f"上传失败：{file_path}")
        return payload[0]

    def call_endpoint(self, api_name: str, data_list: list) -> list:
        call_url = f"{self.server_url}/gradio_api/call/{api_name}"
        response = requests.post(call_url, json={"data": data_list}, timeout=self.timeout)
        response.raise_for_status()
        event_id = response.json().get("event_id")
        if not event_id:
            raise StereoApiError("接口未返回 event_id。")

        stream_url = f"{self.server_url}/gradio_api/call/{api_name}/{event_id}"
        deadline = time.time() + max(self.timeout, 120.0)
        latest_payload = None

        while time.time() < deadline:
            stream_response = requests.get(stream_url, timeout=max(self.timeout, 60.0))
            stream_response.raise_for_status()

            for raw_line in stream_response.text.splitlines():
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if not body:
                    continue
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, list):
                    latest_payload = payload

            if "event: complete" in stream_response.text and latest_payload is not None:
                return latest_payload
            time.sleep(0.5)

        raise StereoApiError("等待模型结果超时。")

    def run_prediction(
        self,
        model_key: str,
        left_image: str | Path,
        right_image: str | Path,
        intrinsic_text: str,
        params: dict,
    ) -> dict:
        left_token = self.upload_file(left_image)
        right_token = self.upload_file(right_image)

        if model_key == "s2m2":
            api_name = "run_s2m2_pipeline"
            payload = [
                {"path": left_token, "meta": {"_type": "gradio.FileData"}},
                {"path": right_token, "meta": {"_type": "gradio.FileData"}},
                intrinsic_text,
                params.get("model_choice", "X-Large (XL)"),
                params.get("get_pc", True),
                params.get("z_far", 10.0),
                params.get("conf_thresh", 0.0),
                params.get("denoise", False),
                params.get("d_nb", 30),
                params.get("d_rad", 0.03),
            ]
        else:
            api_name = "run_official_pipeline"
            payload = [
                {"path": left_token, "meta": {"_type": "gradio.FileData"}},
                {"path": right_token, "meta": {"_type": "gradio.FileData"}},
                intrinsic_text,
                params.get("scale", 1.0),
                params.get("iters", 32),
                params.get("hiera", False),
                params.get("get_pc", True),
                params.get("z_far", 10.0),
                params.get("rm_inv", False),
                params.get("denoise", False),
                params.get("d_nb", 30),
                params.get("d_rad", 0.03),
            ]

        result = self.call_endpoint(api_name, payload)
        if model_key == "s2m2":
            return {
                "disp_vis": result[0],
                "conf_vis": result[1],
                "disp_png": result[2],
                "disp_npy": result[3],
                "conf_npy": result[4],
                "depth_npy": result[5],
                "pointcloud": result[6] if len(result) > 6 else None,
            }
        return {
            "disp_vis": result[0],
            "disp_png": result[1],
            "disp_npy": result[2],
            "depth_npy": result[3],
            "pointcloud": result[4] if len(result) > 4 else None,
        }

    def save_remote_output(self, remote_value, destination: str | Path) -> str:
        source_path = self._extract_output_path(remote_value)
        if not source_path:
            return ""

        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if Path(source_path).exists():
            shutil.copy2(source_path, destination_path)
            return str(destination_path)

        encoded_path = quote(str(source_path), safe="/:\\")
        download_url = f"{self.server_url}/gradio_api/file={encoded_path}"
        response = requests.get(download_url, stream=True, timeout=max(self.timeout, 60.0))
        response.raise_for_status()
        with destination_path.open("wb") as file:
            shutil.copyfileobj(response.raw, file)
        return str(destination_path)

    @staticmethod
    def _extract_output_path(remote_value) -> str:
        if isinstance(remote_value, dict):
            return str(remote_value.get("path") or "")
        if isinstance(remote_value, str):
            return remote_value
        return ""

