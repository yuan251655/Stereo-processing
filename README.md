# Stereo Processing

这是一个聚焦双目深度处理的桌面工具工程，目标是对输入目录中的原始图像执行自动参数匹配、图像校正和深度推理，并统一产出：

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

1. 在 `ui.py` 里设置输入目录与输出目录。
2. 选择深度模型，并确认服务地址：
   - `预处理服务` 对应 `http://192.168.30.237:8002`
   - `Foundation Stereo` 对应 `http://192.168.30.237:8000`
   - `S2M2 Stereo Matching` 对应 `http://192.168.30.237:8001`
3. 根据模型调整参数。
4. 点“检查服务”确认可用后，直接点“开始处理”。
5. 程序会自动扫描输入目录中的原始图像，逐个完成参数匹配、校正、深度推理与结果保存。
6. 处理完成后，到右侧查看结果记录，双击输出文件可打开所在目录。

## 界面布局说明

- 当前桌面界面采用“主配置区 + 辅助预览结果区”的布局，目录、模型、参数、执行控制会优先占据主要操作区域。
- 配置卡片会根据可用宽度自动在双列和单列之间切换，避免缩小时表单挤成一条窄长栏。
- 预览区改为单主画布标签切换，分别查看左图、右图、视差结果和辅助结果，缩小时不会再把多块预览同时压扁。
- 辅助区整体采用统一纵向滚动，预览、日志、结果记录、输出文件和结果摘要不要求一屏同时塞满，窗口缩小时优先保证每张卡片自身可读。
- 当窗口宽度不足时，主分栏会自动改为上下堆叠，优先保证文字可读性和控件可操作性。

## 注意事项

- 当前工程默认依赖 `PyQt5`、`requests`、`PyYAML`、`gradio_client`。
- 处理流程依赖预处理服务自动查询相机参数，并生成左右校正图与新的内参与基线。
- 输入目录根目录应放置原始待处理图像；运行后会在 `input/left` 和 `input/right` 中缓存对应样本的左右校正图，并在 `input/pairs` 生成样本清单。
- `output/metadata/*.json` 会记录每次任务的输入、参数、模型、耗时和产物路径，方便后续追踪。
