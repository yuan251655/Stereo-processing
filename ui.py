from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from src.app_config import ensure_default_config, load_config, save_config
from src.file_store import list_saved_results
from src.pipeline_runner import PipelineRunner
from src.stereo_api import StereoApiClient

try:
    from PyQt5.QtCore import QThread, Qt, pyqtSignal
    from PyQt5.QtGui import QFont, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSplitter,
        QStackedWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("缺少 PyQt5，请先安装：pip install PyQt5")
    raise SystemExit(1)


BASE_DIR = Path(__file__).resolve().parent


class PreviewPane(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setObjectName("Card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("CardTitle")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(220)
        self.image_label.setObjectName("PreviewSurface")
        self.image_label.setWordWrap(True)
        self.caption_label = QLabel("等待内容")
        self.caption_label.setObjectName("HintText")
        self.caption_label.setWordWrap(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.caption_label)
        self.set_empty("暂无内容", "选择输入图像或结果后，这里会显示预览。")

    def set_empty(self, title: str, detail: str) -> None:
        self._pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(title)
        self.caption_label.setText(detail)

    def set_image(self, image_path: str, detail: str = "") -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.set_empty("无法加载预览", image_path)
            return
        self._pixmap = pixmap
        self.caption_label.setText(detail or image_path)
        self._refresh_pixmap()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)


class PipelineThread(QThread):
    log_signal = pyqtSignal(str)
    stage_signal = pyqtSignal(str, str)
    progress_signal = pyqtSignal(int, int)
    completed_signal = pyqtSignal(bool, str, object)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._stop_requested = False

    def run(self) -> None:
        runner = PipelineRunner(
            self.config,
            logger=self.log_signal.emit,
            status_callback=self.stage_signal.emit,
            progress_callback=self.progress_signal.emit,
            stop_flag=lambda: self._stop_requested,
        )
        result = runner.run()
        self.completed_signal.emit(result.success, result.message, result.metadata)

    def stop(self) -> None:
        self._stop_requested = True


class StereoMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_default_config()
        self.config = load_config()
        self.worker = None
        self.result_records = []
        self.current_record = None

        self.setWindowTitle("Stereo Processing Workbench")
        self.resize(1680, 980)
        self.setMinimumSize(1360, 820)
        self.setFont(QFont("Microsoft YaHei UI", 10))

        self._build_ui()
        self._apply_styles()
        self.load_config_to_widgets()
        self.refresh_results()
        self.refresh_live_preview()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        root_layout.addWidget(self._build_header())

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self._build_left_panel())
        main_splitter.addWidget(self._build_center_panel())
        main_splitter.addWidget(self._build_right_panel())
        main_splitter.setSizes([420, 820, 360])
        root_layout.addWidget(main_splitter, stretch=1)
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        card = QFrame()
        card.setObjectName("HeroCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(18)

        text_column = QVBoxLayout()
        text_column.setSpacing(4)
        title = QLabel("Stereo Processing Workbench")
        title.setObjectName("HeroTitle")
        subtitle = QLabel("聚焦双目视差、深度与点云产出，不再包含苗高、分割或 YOLO 相关流程。")
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        text_column.addWidget(title)
        text_column.addWidget(subtitle)

        badge_column = QVBoxLayout()
        badge_column.setSpacing(8)
        badge_column.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_chip = QLabel("就绪")
        self.status_chip.setObjectName("StatusChip")
        self.status_value = QLabel("等待开始")
        self.status_value.setObjectName("StatusValue")
        badge_column.addWidget(self.status_chip, alignment=Qt.AlignRight)
        badge_column.addWidget(self.status_value, alignment=Qt.AlignRight)

        layout.addLayout(text_column, stretch=1)
        layout.addLayout(badge_column)
        return card

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(14)
        layout.addWidget(self._build_workspace_card())
        layout.addWidget(self._build_source_card())
        layout.addWidget(self._build_model_card())
        layout.addWidget(self._build_params_card())
        layout.addWidget(self._build_action_card())
        layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)
        return container

    def _build_workspace_card(self) -> QWidget:
        card, layout = self._card_with_layout("工作区与目录", "固定 input / output 目录职责，结果按类型自动分类。")
        self.input_dir_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self._add_path_row(layout, "输入目录", self.input_dir_edit, self.select_input_dir)
        self._add_path_row(layout, "输出目录", self.output_dir_edit, self.select_output_dir)
        return card

    def _build_source_card(self) -> QWidget:
        card, layout = self._card_with_layout("当前样本", "选择本次处理的左右图像，并写入 input 目录备份。")
        self.sample_name_edit = QLineEdit()
        self.left_image_edit = QLineEdit()
        self.right_image_edit = QLineEdit()
        self.intrinsic_edit = QTextEdit()
        self.intrinsic_edit.setMinimumHeight(92)
        self.copy_input_checkbox = QCheckBox("运行时复制原始左右图到 input 目录")
        self.left_image_edit.textChanged.connect(self.refresh_live_preview)
        self.right_image_edit.textChanged.connect(self.refresh_live_preview)
        self._add_form_row(layout, "样本名", self.sample_name_edit)
        self._add_path_row(layout, "左图路径", self.left_image_edit, self.select_left_image)
        self._add_path_row(layout, "右图路径", self.right_image_edit, self.select_right_image)
        self._add_form_row(layout, "内参 / 基线", self.intrinsic_edit)
        layout.addWidget(self.copy_input_checkbox)
        return card

    def _build_model_card(self) -> QWidget:
        card, layout = self._card_with_layout("模型与服务", "支持 FS 与 S2M2 两套接口，状态检查走你提供的官方 API。")
        self.model_combo = QComboBox()
        self.model_combo.addItem("Foundation Stereo", "fs")
        self.model_combo.addItem("S2M2 Stereo Matching", "s2m2")
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.fs_url_edit = QLineEdit()
        self.s2m2_url_edit = QLineEdit()
        self.service_status_label = QLabel("尚未检查服务")
        self.service_status_label.setObjectName("HintText")
        self.service_status_label.setWordWrap(True)
        self._add_form_row(layout, "当前模型", self.model_combo)
        self._add_form_row(layout, "FS 服务地址", self.fs_url_edit)
        self._add_form_row(layout, "S2M2 服务地址", self.s2m2_url_edit)
        layout.addWidget(self.service_status_label)
        return card

    def _build_params_card(self) -> QWidget:
        card, layout = self._card_with_layout("参数配置", "按模型切换对应参数，尽量保持可读和直观。")
        self.params_stack = QStackedWidget()
        self.fs_param_widgets = {}
        self.s2m2_param_widgets = {}
        self.params_stack.addWidget(self._build_fs_params_page())
        self.params_stack.addWidget(self._build_s2m2_params_page())
        layout.addWidget(self.params_stack)
        return card

    def _build_fs_params_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        for key in ["scale", "iters", "z_far", "d_nb", "d_rad"]:
            self.fs_param_widgets[key] = QLineEdit()
        self.fs_param_widgets["hiera"] = QCheckBox("开启分层推理")
        self.fs_param_widgets["get_pc"] = QCheckBox("生成点云 .ply")
        self.fs_param_widgets["rm_inv"] = QCheckBox("移除不可见点")
        self.fs_param_widgets["denoise"] = QCheckBox("点云去噪")
        rows = [("缩放比例", "scale"), ("迭代次数", "iters"), ("Z-Far", "z_far"), ("去噪邻域点数", "d_nb"), ("去噪半径", "d_rad")]
        for row, (label_text, key) in enumerate(rows):
            layout.addWidget(QLabel(label_text), row, 0)
            layout.addWidget(self.fs_param_widgets[key], row, 1)
        row = len(rows)
        for offset, key in enumerate(["hiera", "get_pc", "rm_inv", "denoise"]):
            layout.addWidget(self.fs_param_widgets[key], row + offset, 0, 1, 2)
        return page

    def _build_s2m2_params_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)

        model_choice = QComboBox()
        for item in ["Small (S)", "Medium (M)", "Large (L)", "X-Large (XL)"]:
            model_choice.addItem(item)

        self.s2m2_param_widgets["model_choice"] = model_choice
        for key in ["z_far", "conf_thresh", "d_nb", "d_rad"]:
            self.s2m2_param_widgets[key] = QLineEdit()
        self.s2m2_param_widgets["get_pc"] = QCheckBox("生成点云 .ply")
        self.s2m2_param_widgets["denoise"] = QCheckBox("点云去噪")

        rows = [
            ("模型规模", "model_choice"),
            ("Z-Far", "z_far"),
            ("置信阈值", "conf_thresh"),
            ("去噪邻域点数", "d_nb"),
            ("去噪半径", "d_rad"),
        ]
        for row, (label_text, key) in enumerate(rows):
            layout.addWidget(QLabel(label_text), row, 0)
            layout.addWidget(self.s2m2_param_widgets[key], row, 1)
        row = len(rows)
        layout.addWidget(self.s2m2_param_widgets["get_pc"], row, 0, 1, 2)
        layout.addWidget(self.s2m2_param_widgets["denoise"], row + 1, 0, 1, 2)
        return page

    def _build_action_card(self) -> QWidget:
        card, layout = self._card_with_layout("执行控制", "先保存配置，再检查服务，最后启动处理任务。")
        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self.save_button = QPushButton("保存配置")
        self.check_button = QPushButton("检查服务")
        self.run_button = QPushButton("开始处理")
        self.stop_button = QPushButton("停止")
        self.open_output_button = QPushButton("打开输出目录")

        self.save_button.clicked.connect(self.save_current_config)
        self.check_button.clicked.connect(self.check_current_service)
        self.run_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)
        self.open_output_button.clicked.connect(self.open_output_dir)
        self.stop_button.setEnabled(False)

        for button in [self.save_button, self.check_button, self.run_button, self.stop_button, self.open_output_button]:
            button_row.addWidget(button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_hint = QLabel("等待开始")
        self.progress_hint.setObjectName("HintText")

        layout.addLayout(button_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_hint)
        return card

    def _build_center_panel(self) -> QWidget:
        splitter = QSplitter(Qt.Vertical)

        preview_card = QFrame()
        preview_card.setObjectName("Card")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.setSpacing(12)
        title = QLabel("预览工作台")
        title.setObjectName("CardTitle")
        subtitle = QLabel("上排关注输入，下排关注输出与辅助结果。")
        subtitle.setObjectName("HintText")
        subtitle.setWordWrap(True)
        preview_layout.addWidget(title)
        preview_layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self.left_preview = PreviewPane("左图")
        self.right_preview = PreviewPane("右图")
        self.result_preview = PreviewPane("视差可视化")
        self.aux_preview = PreviewPane("辅助结果")
        grid.addWidget(self.left_preview, 0, 0)
        grid.addWidget(self.right_preview, 0, 1)
        grid.addWidget(self.result_preview, 1, 0)
        grid.addWidget(self.aux_preview, 1, 1)
        preview_layout.addLayout(grid, stretch=1)

        log_card = QFrame()
        log_card.setObjectName("Card")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.setSpacing(12)
        log_title = QLabel("运行日志")
        log_title.setObjectName("CardTitle")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(220)
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log_output, stretch=1)

        splitter.addWidget(preview_card)
        splitter.addWidget(log_card)
        splitter.setSizes([640, 260])
        return splitter

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        history_card, history_layout = self._card_with_layout("结果记录", "按最近处理时间倒序展示，可快速回看每次输出。")
        self.results_list = QListWidget()
        self.results_list.itemSelectionChanged.connect(self.on_result_selected)
        history_layout.addWidget(self.results_list)

        file_card, file_layout = self._card_with_layout("输出文件", "查看当前结果包含哪些产物，双击可打开所在位置。")
        self.output_files_list = QListWidget()
        self.output_files_list.itemDoubleClicked.connect(self.open_selected_artifact)
        file_layout.addWidget(self.output_files_list)

        meta_card, meta_layout = self._card_with_layout("结果摘要", "记录模型、耗时、参数与路径，便于回查。")
        self.metadata_view = QTextEdit()
        self.metadata_view.setReadOnly(True)
        self.metadata_view.setMinimumHeight(220)
        meta_layout.addWidget(self.metadata_view)

        layout.addWidget(history_card, stretch=2)
        layout.addWidget(file_card, stretch=2)
        layout.addWidget(meta_card, stretch=3)
        return container

    def _card_with_layout(self, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        title_label = QLabel(title)
        title_label.setObjectName("CardTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("HintText")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return card, layout

    def _add_form_row(self, layout: QVBoxLayout, label_text: str, widget: QWidget) -> None:
        label = QLabel(label_text)
        label.setObjectName("FieldLabel")
        layout.addWidget(label)
        layout.addWidget(widget)

    def _add_path_row(self, layout: QVBoxLayout, label_text: str, line_edit: QLineEdit, browse_handler) -> None:
        label = QLabel(label_text)
        label.setObjectName("FieldLabel")
        row = QHBoxLayout()
        row.setSpacing(8)
        button = QPushButton("浏览")
        button.setProperty("small", True)
        button.clicked.connect(browse_handler)
        row.addWidget(line_edit, stretch=1)
        row.addWidget(button)
        layout.addWidget(label)
        layout.addLayout(row)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #eef3f8; color: #17212f; }
            #HeroCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f5d91, stop:1 #0b3c5d);
                border-radius: 22px;
            }
            #HeroTitle { color: white; font-size: 24px; font-weight: 700; }
            #HeroSubtitle { color: rgba(255,255,255,0.86); font-size: 13px; }
            #StatusChip {
                background: rgba(255,255,255,0.18); color: white; border-radius: 14px; padding: 7px 14px; font-weight: 700;
            }
            #StatusValue { color: rgba(255,255,255,0.88); font-size: 12px; }
            #Card { background: #fbfdff; border: 1px solid #d8e3ef; border-radius: 18px; }
            #CardTitle { font-size: 16px; font-weight: 700; color: #17324d; }
            #FieldLabel { font-size: 12px; font-weight: 600; color: #43607c; margin-top: 4px; }
            #HintText { font-size: 12px; color: #637a92; }
            #PreviewSurface {
                background: #edf3f9; border: 1px dashed #bdd0e2; border-radius: 14px; color: #5d7289; padding: 12px;
            }
            QLineEdit, QTextEdit, QComboBox, QListWidget {
                background: white; border: 1px solid #c8d5e3; border-radius: 12px; padding: 10px 12px; selection-background-color: #0f5d91;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QListWidget:focus { border: 1px solid #0f5d91; }
            QPushButton {
                background: #0f5d91; border: none; color: white; border-radius: 12px; padding: 10px 16px; font-weight: 600;
            }
            QPushButton:hover { background: #0c4e79; }
            QPushButton[small="true"] { padding: 8px 12px; border-radius: 10px; }
            QPushButton:disabled { background: #9cb3c7; color: #edf3f8; }
            QCheckBox { spacing: 8px; color: #2c4359; }
            QProgressBar {
                background: #e6eef6; border: none; border-radius: 10px; min-height: 16px; text-align: center;
            }
            QProgressBar::chunk { background: #2f8f68; border-radius: 10px; }
            QSplitter::handle { background: transparent; }
            """
        )

    def load_config_to_widgets(self) -> None:
        config = self.config
        self.input_dir_edit.setText(config["paths"]["input_dir"])
        self.output_dir_edit.setText(config["paths"]["output_dir"])
        self.sample_name_edit.setText(config["session"]["sample_name"])
        self.left_image_edit.setText(config["session"]["left_image"])
        self.right_image_edit.setText(config["session"]["right_image"])
        self.intrinsic_edit.setPlainText(config["session"]["intrinsic_text"])
        self.copy_input_checkbox.setChecked(bool(config["session"]["save_input_copy"]))
        self.fs_url_edit.setText(config["servers"]["fs"])
        self.s2m2_url_edit.setText(config["servers"]["s2m2"])
        model_key = config.get("model_selection", "fs")
        self.model_combo.setCurrentIndex(0 if model_key == "fs" else 1)

        fs = config["fs_params"]
        self.fs_param_widgets["scale"].setText(str(fs["scale"]))
        self.fs_param_widgets["iters"].setText(str(fs["iters"]))
        self.fs_param_widgets["z_far"].setText(str(fs["z_far"]))
        self.fs_param_widgets["d_nb"].setText(str(fs["d_nb"]))
        self.fs_param_widgets["d_rad"].setText(str(fs["d_rad"]))
        self.fs_param_widgets["hiera"].setChecked(bool(fs["hiera"]))
        self.fs_param_widgets["get_pc"].setChecked(bool(fs["get_pc"]))
        self.fs_param_widgets["rm_inv"].setChecked(bool(fs["rm_inv"]))
        self.fs_param_widgets["denoise"].setChecked(bool(fs["denoise"]))

        s2m2 = config["s2m2_params"]
        self.s2m2_param_widgets["model_choice"].setCurrentText(str(s2m2["model_choice"]))
        self.s2m2_param_widgets["z_far"].setText(str(s2m2["z_far"]))
        self.s2m2_param_widgets["conf_thresh"].setText(str(s2m2["conf_thresh"]))
        self.s2m2_param_widgets["d_nb"].setText(str(s2m2["d_nb"]))
        self.s2m2_param_widgets["d_rad"].setText(str(s2m2["d_rad"]))
        self.s2m2_param_widgets["get_pc"].setChecked(bool(s2m2["get_pc"]))
        self.s2m2_param_widgets["denoise"].setChecked(bool(s2m2["denoise"]))
        self.on_model_changed()

    def collect_form_data(self) -> dict:
        config = load_config()
        config["paths"]["input_dir"] = self.input_dir_edit.text().strip()
        config["paths"]["output_dir"] = self.output_dir_edit.text().strip()
        config["session"]["sample_name"] = self.sample_name_edit.text().strip()
        config["session"]["left_image"] = self.left_image_edit.text().strip()
        config["session"]["right_image"] = self.right_image_edit.text().strip()
        config["session"]["intrinsic_text"] = self.intrinsic_edit.toPlainText().strip()
        config["session"]["save_input_copy"] = self.copy_input_checkbox.isChecked()
        config["servers"]["fs"] = self.fs_url_edit.text().strip()
        config["servers"]["s2m2"] = self.s2m2_url_edit.text().strip()
        config["model_selection"] = self.model_combo.currentData()
        config["fs_params"] = {
            "scale": float(self.fs_param_widgets["scale"].text().strip() or 1.0),
            "iters": int(float(self.fs_param_widgets["iters"].text().strip() or 32)),
            "hiera": self.fs_param_widgets["hiera"].isChecked(),
            "get_pc": self.fs_param_widgets["get_pc"].isChecked(),
            "z_far": float(self.fs_param_widgets["z_far"].text().strip() or 10.0),
            "rm_inv": self.fs_param_widgets["rm_inv"].isChecked(),
            "denoise": self.fs_param_widgets["denoise"].isChecked(),
            "d_nb": int(float(self.fs_param_widgets["d_nb"].text().strip() or 30)),
            "d_rad": float(self.fs_param_widgets["d_rad"].text().strip() or 0.03),
        }
        config["s2m2_params"] = {
            "model_choice": self.s2m2_param_widgets["model_choice"].currentText(),
            "get_pc": self.s2m2_param_widgets["get_pc"].isChecked(),
            "z_far": float(self.s2m2_param_widgets["z_far"].text().strip() or 10.0),
            "conf_thresh": float(self.s2m2_param_widgets["conf_thresh"].text().strip() or 0.0),
            "denoise": self.s2m2_param_widgets["denoise"].isChecked(),
            "d_nb": int(float(self.s2m2_param_widgets["d_nb"].text().strip() or 30)),
            "d_rad": float(self.s2m2_param_widgets["d_rad"].text().strip() or 0.03),
        }
        return config

    def save_current_config(self) -> None:
        try:
            self.config = self.collect_form_data()
            save_config(self.config)
            self.append_log("配置已保存到 config.yaml")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", f"配置校验失败：\n{exc}")

    def on_model_changed(self) -> None:
        model_key = self.model_combo.currentData()
        self.params_stack.setCurrentIndex(0 if model_key == "fs" else 1)
        self.service_status_label.setText(
            "当前接口：/get_system_status + /run_official_pipeline"
            if model_key == "fs"
            else "当前接口：/get_status + /run_s2m2_pipeline"
        )

    def refresh_live_preview(self) -> None:
        left_path = self.left_image_edit.text().strip()
        right_path = self.right_image_edit.text().strip()
        if Path(left_path).exists():
            self.left_preview.set_image(left_path, left_path)
        else:
            self.left_preview.set_empty("左图待选择", "请选择一张左图。")
        if Path(right_path).exists():
            self.right_preview.set_image(right_path, right_path)
        else:
            self.right_preview.set_empty("右图待选择", "请选择一张右图。")

    def start_processing(self) -> None:
        try:
            self.config = self.collect_form_data()
            save_config(self.config)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "无法开始", f"参数填写有误：\n{exc}")
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "任务进行中", "当前已有任务在运行，请先等待完成。")
            return

        self.worker = PipelineThread(self.config, self)
        self.worker.log_signal.connect(self.append_log)
        self.worker.stage_signal.connect(self.update_stage)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.completed_signal.connect(self.on_processing_finished)

        self.log_output.clear()
        self.append_log("开始处理任务...")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_hint.setText("模型处理中...")
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_chip.setText("运行中")
        self.status_value.setText("正在调用模型服务")
        self.worker.start()

    def stop_processing(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.append_log("已请求停止，等待当前网络请求结束...")
            self.stop_button.setEnabled(False)

    def on_processing_finished(self, success: bool, message: str, metadata) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_chip.setText("完成" if success else "失败")
        self.status_value.setText(message)
        self.progress_hint.setText(message)
        self.append_log(message)

        if success and metadata:
            self.refresh_results(select_sample=metadata.get("sample_name"))
        else:
            self.refresh_results()

        if not success:
            QMessageBox.warning(self, "处理结束", message)

    def update_stage(self, stage: str, sample_name: str) -> None:
        stage_map = {"connecting": "连接服务", "running": "执行推理", "finished": "写入结果"}
        text = stage_map.get(stage, stage)
        self.status_value.setText(f"{text} · {sample_name}")
        self.append_log(f"[阶段] {text} - {sample_name}")

    def update_progress(self, current: int, total: int) -> None:
        total = max(total, 1)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.progress_hint.setText(f"进度 {current}/{total}")

    def append_log(self, message: str) -> None:
        self.log_output.append(message)

    def refresh_results(self, select_sample: str | None = None) -> None:
        output_dir = self.output_dir_edit.text().strip() or self.config["paths"]["output_dir"]
        self.result_records = list_saved_results(output_dir)
        self.results_list.clear()

        selected_item = None
        for record in self.result_records:
            label = f"{record.get('sample_name', 'unknown')} · {record.get('model', '').upper()}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, record)
            self.results_list.addItem(item)
            if select_sample and record.get("sample_name") == select_sample:
                selected_item = item

        if self.results_list.count() == 0:
            self.metadata_view.setPlainText("暂无处理结果。")
            self.output_files_list.clear()
            self.result_preview.set_empty("暂无结果", "运行一次任务后，这里会显示视差结果。")
            self.aux_preview.set_empty("暂无辅助产物", "S2M2 置信图或点云文件信息会显示在这里。")
            return

        if selected_item is None:
            selected_item = self.results_list.item(0)
        self.results_list.setCurrentItem(selected_item)

    def on_result_selected(self) -> None:
        item = self.results_list.currentItem()
        if not item:
            return

        record = item.data(Qt.UserRole)
        self.current_record = record
        outputs = record.get("outputs", {})
        inputs = record.get("inputs", {})

        left_path = inputs.get("left_image", "")
        right_path = inputs.get("right_image", "")
        if left_path and Path(left_path).exists():
            self.left_preview.set_image(left_path, left_path)
        if right_path and Path(right_path).exists():
            self.right_preview.set_image(right_path, right_path)

        disp_vis = outputs.get("disp_vis", "")
        if disp_vis and Path(disp_vis).exists():
            self.result_preview.set_image(disp_vis, disp_vis)
        else:
            self.result_preview.set_empty("缺少视差可视化", "当前记录没有可显示的视差可视化图。")

        aux_path = outputs.get("conf_vis") or outputs.get("disp_png")
        if aux_path and Path(aux_path).exists():
            self.aux_preview.set_image(aux_path, aux_path)
        else:
            pointcloud_path = outputs.get("pointcloud", "")
            if pointcloud_path:
                self.aux_preview.set_empty("点云文件已生成", pointcloud_path)
            else:
                self.aux_preview.set_empty("暂无辅助结果", "当前记录没有置信图或点云。")

        self.output_files_list.clear()
        for key, value in outputs.items():
            if not value:
                continue
            file_item = QListWidgetItem(f"{key}: {value}")
            file_item.setData(Qt.UserRole, value)
            self.output_files_list.addItem(file_item)

        pretty_record = {
            "sample_name": record.get("sample_name"),
            "model": record.get("model"),
            "created_at": record.get("created_at"),
            "elapsed_seconds": record.get("elapsed_seconds"),
            "server_url": record.get("server_url"),
            "parameters": record.get("parameters"),
            "metadata_path": record.get("metadata_path"),
        }
        self.metadata_view.setPlainText(json.dumps(pretty_record, ensure_ascii=False, indent=2))

    def open_selected_artifact(self, item: QListWidgetItem) -> None:
        file_path = item.data(Qt.UserRole)
        if file_path:
            target = Path(file_path)
            if target.exists():
                self.open_in_system(target)

    def check_current_service(self) -> None:
        try:
            config = self.collect_form_data()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "检查失败", f"请先修正配置：\n{exc}")
            return

        model_key = config["model_selection"]
        server_url = config["servers"][model_key]
        self.service_status_label.setText("正在检查服务，请稍候...")
        QApplication.processEvents()

        try:
            client = StereoApiClient(server_url, timeout=10)
            status_text, timer_text = client.get_status(model_key)
            self.service_status_label.setText(f"服务正常：{status_text} | {timer_text}")
            self.append_log(f"服务检查成功：{server_url}")
        except Exception as exc:  # noqa: BLE001
            self.service_status_label.setText(f"服务检查失败：{exc}")
            self.append_log(f"服务检查失败：{server_url} -> {exc}")

    def select_input_dir(self) -> None:
        current = self.input_dir_edit.text().strip() or str(BASE_DIR / "input")
        folder = QFileDialog.getExistingDirectory(self, "选择输入目录", current)
        if folder:
            self.input_dir_edit.setText(folder)

    def select_output_dir(self) -> None:
        current = self.output_dir_edit.text().strip() or str(BASE_DIR / "output")
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", current)
        if folder:
            self.output_dir_edit.setText(folder)
            self.refresh_results()

    def select_left_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择左图", str(BASE_DIR), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if file_path:
            self.left_image_edit.setText(file_path)
            self._autofill_sample_name(file_path)

    def select_right_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择右图", str(BASE_DIR), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if file_path:
            self.right_image_edit.setText(file_path)
            if not self.sample_name_edit.text().strip():
                self._autofill_sample_name(file_path)

    def _autofill_sample_name(self, file_path: str) -> None:
        if not self.sample_name_edit.text().strip():
            self.sample_name_edit.setText(Path(file_path).stem)

    def open_output_dir(self) -> None:
        self.open_in_system(Path(self.output_dir_edit.text().strip()))

    @staticmethod
    def open_in_system(path: Path) -> None:
        if not path:
            return
        target = path if path.is_dir() else path.parent
        if not target.exists():
            return
        if sys.platform.startswith("win"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f"open \"{target}\"")
        else:
            os.system(f"xdg-open \"{target}\"")


def main() -> int:
    app = QApplication(sys.argv)
    window = StereoMainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
