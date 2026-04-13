# Stereo Processing

这是一个聚焦双目深度处理的桌面工具工程，目标是把输入双目图像送入两套远端模型服务，统一产出：

- 视差可视化图 `disparity_vis/*.png`
- 视差图文件 `disparity_png/*.png`
- 视差原始数据 `disparity_npy/*.npy`
- 深度数据 `depth_npy/*.npy`
- 点云文件 `pointcloud/*.ply`
- S2M2 置信度结果 `confidence_vis/*.png` / `confidence_npy/*.npy`
- 结果元数据 `metadata/*.json`

工程不再包含苗高计算、分割、YOLO 推理或相关后处理。

## 目录结构

```text
Stereo-processing/
├─ config.yaml
├─ process_pipeline.py
├─ ui.py
├─ input/
│  ├─ left/
│  ├─ right/
│  └─ pairs/
├─ output/
│  ├─ disparity_vis/
│  ├─ disparity_png/
│  ├─ disparity_npy/
│  ├─ depth_npy/
│  ├─ pointcloud/
│  ├─ confidence_vis/
│  ├─ confidence_npy/
│  └─ metadata/
└─ src/
   ├─ app_config.py
   ├─ file_store.py
   ├─ pipeline_runner.py
   └─ stereo_api.py
```

## 运行方式

直接启动桌面界面：

```bash
python ui.py
```

如果你只想按当前 `config.yaml` 直接跑一次后台流程，也可以执行：

```bash
python process_pipeline.py
```

## 使用说明

1. 在 `ui.py` 里填写左右图路径、样本名、内参与基线。
2. 选择模型：
   - `Foundation Stereo` 对应 `http://192.168.30.237:8000`
   - `S2M2 Stereo Matching` 对应 `http://192.168.30.237:8001`
3. 根据模型调整参数。
4. 先点“保存配置”，再点“检查服务”，最后点“开始处理”。
5. 处理完成后，到右侧查看结果记录，双击输出文件可打开所在目录。

## 注意事项

- 当前工程默认依赖 `PyQt5`、`requests`、`PyYAML`、`gradio_client`。
- 两套模型 API 都要求输入为左右图像和一段内参与基线文本。
- 运行时如果勾选“复制原始左右图到 input 目录”，原始图像会分别写入 `input/left` 和 `input/right`，并在 `input/pairs` 生成对应样本清单。
- `output/metadata/*.json` 会记录每次任务的输入、参数、模型、耗时和产物路径，方便后续追踪。
