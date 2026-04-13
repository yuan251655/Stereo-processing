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
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QSizePolicy,
    )
except ImportError:
    print("缺少 PyQt5，请先安装：pip install PyQt5")
    raise SystemExit(1)


BASE_DIR = Path(__file__).resolve().parent


class PreviewPane(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._caption_text = ""
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("InlineLabel")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(220)
        self.image_label.setObjectName("PreviewSurface")
        self.image_label.setWordWrap(True)
        self.caption_label = QLabel("等待内容")
        self.caption_label.setObjectName("PathCaption")
        self.caption_label.setWordWrap(False)
        self.caption_label.setMinimumHeight(18)

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.caption_label)
        self.set_empty("暂无内容", "选择输入图像或结果后，这里会显示预览。")

    def set_empty(self, title: str, detail: str) -> None:
        self._pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(title)
        self._caption_text = detail
        self._refresh_caption()

    def set_image(self, image_path: str, detail: str = "") -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.set_empty("无法加载预览", image_path)
            return
        self._pixmap = pixmap
        self._caption_text = detail or image_path
        self._refresh_caption()
        self._refresh_pixmap()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()
        self._refresh_caption()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _refresh_caption(self) -> None:
        if not self._caption_text:
            self.caption_label.clear()
            return
        width = max(self.caption_label.width() - 6, 120)
        text = self.caption_label.fontMetrics().elidedText(self._caption_text, Qt.ElideMiddle, width)
        self.caption_label.setText(text)

class AdaptiveConfigGrid(QWidget):
    def __init__(self, cards: dict[str, QWidget], parent=None):
        super().__init__(parent)
        self._cards = cards
        self._compact = None
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(16)
        self._grid.setVerticalSpacing(16)
        self._apply_layout(False)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        compact = self.width() < 900
        if compact != self._compact:
            self._apply_layout(compact)

    def _apply_layout(self, compact: bool) -> None:
        self._compact = compact
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                self._grid.removeWidget(widget)

        for column in range(2):
            self._grid.setColumnStretch(column, 0)
        for row in range(5):
            self._grid.setRowStretch(row, 0)

        if compact:
            order = ["source", "workspace", "model", "params", "action"]
            for row, key in enumerate(order):
                self._grid.addWidget(self._cards[key], row, 0)
            self._grid.setColumnStretch(0, 1)
            return

        self._grid.addWidget(self._cards["source"], 0, 0, 1, 2)
        self._grid.addWidget(self._cards["workspace"], 1, 0)
        self._grid.addWidget(self._cards["model"], 1, 1)
        self._grid.addWidget(self._cards["params"], 2, 0, 1, 2)
        self._grid.addWidget(self._cards["action"], 3, 0, 1, 2)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 1)


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
        self._compact_window = None

        self.setWindowTitle("Stereo Processing Workbench")
        self.resize(1560, 960)
        self.setMinimumSize(980, 720)
        self.setFont(QFont("Microsoft YaHei UI", 11))

        self._build_ui()
        self._apply_styles()
        self.load_config_to_widgets()
        self.refresh_results()
        self.refresh_live_preview()
        self._set_status("ready", "就绪", "等待开始")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(16)

        root_layout.addWidget(self._build_header())

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(10)

        left_panel = self._build_left_panel()
        side_panel = self._build_side_panel()
        left_panel.setMinimumWidth(560)
        side_panel.setMinimumWidth(380)

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(side_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([940, 620])
        root_layout.addWidget(self.main_splitter, stretch=1)
        self.setCentralWidget(root)
        self._update_responsive_layout(force=True)

    def _build_header(self) -> QWidget:
        card = QFrame()
        card.setObjectName("ToolbarCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)

        text_column = QVBoxLayout()
        text_column.setSpacing(4)
        eyebrow = QLabel("STEREO PROCESSING")
        eyebrow.setObjectName("ToolbarEyebrow")
        title = QLabel("Stereo Processing Workbench")
        title.setObjectName("ToolbarTitle")
        subtitle = QLabel("核心配置放在主工作区，预览、日志和结果作为辅助区域，整体更适合长时间处理与调参。")
        subtitle.setObjectName("ToolbarSubtitle")
        subtitle.setWordWrap(True)
        text_column.addWidget(eyebrow)
        text_column.addWidget(title)
        text_column.addWidget(subtitle)

        badge_column = QVBoxLayout()
        badge_column.setSpacing(6)
        badge_column.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.setContentsMargins(0, 0, 0, 0)
        status_label = QLabel("当前状态")
        status_label.setObjectName("StatusLabel")
        self.status_chip = QLabel("就绪")
        self.status_chip.setObjectName("StatusChip")
        self.status_value = QLabel("等待开始")
        self.status_value.setObjectName("StatusValue")
        self.status_value.setWordWrap(True)
        self.status_value.setMinimumWidth(0)
        metric_row = QHBoxLayout()
        metric_row.setSpacing(8)
        metric_row.setContentsMargins(0, 0, 0, 0)
        self.model_summary_chip = QLabel("模型 -")
        self.model_summary_chip.setObjectName("MetricPill")
        self.result_count_chip = QLabel("记录 0")
        self.result_count_chip.setObjectName("MetricPill")
        status_row.addWidget(status_label)
        status_row.addWidget(self.status_chip)
        metric_row.addWidget(self.model_summary_chip)
        metric_row.addWidget(self.result_count_chip)
        badge_column.addLayout(status_row)
        badge_column.addWidget(self.status_value, alignment=Qt.AlignRight)
        badge_column.addLayout(metric_row)

        layout.addLayout(text_column, stretch=1)
        layout.addLayout(badge_column)
        return card

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(16)

        cards = {
            "workspace": self._build_workspace_card(),
            "source": self._build_source_card(),
            "model": self._build_model_card(),
            "params": self._build_params_card(),
            "action": self._build_action_card(),
        }
        self.config_grid = AdaptiveConfigGrid(cards)
        layout.addWidget(self.config_grid)
        layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)
        return container

    def _build_workspace_card(self) -> QWidget:
        card, layout = self._card_with_layout("01 工作区", "输入和输出目录固定职责，切换输出目录后会同步刷新右侧结果。")
        self.input_dir_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("默认使用项目下的 input 目录")
        self.output_dir_edit.setPlaceholderText("默认使用项目下的 output 目录")
        self.input_dir_edit.setClearButtonEnabled(True)
        self.output_dir_edit.setClearButtonEnabled(True)
        self._add_path_row(layout, "输入目录", self.input_dir_edit, self.select_input_dir)
        self._add_path_row(layout, "输出目录", self.output_dir_edit, self.select_output_dir)
        return card

    def _build_source_card(self) -> QWidget:
        card, layout = self._card_with_layout("02 样本输入", "先选左右图，再补充样本名和标定信息；运行时可自动备份原图。")
        self.sample_name_edit = QLineEdit()
        self.left_image_edit = QLineEdit()
        self.right_image_edit = QLineEdit()
        self.intrinsic_edit = QTextEdit()
        self.intrinsic_edit.setObjectName("CodeInput")
        self.intrinsic_edit.setMinimumHeight(92)
        self.intrinsic_edit.setPlaceholderText("粘贴相机内参、基线或完整标定文本…")
        self.intrinsic_edit.setAcceptRichText(False)
        self.copy_input_checkbox = QCheckBox("运行时复制原始左右图到 input 目录")
        self.sample_name_edit.setPlaceholderText("例如：sample_001")
        self.left_image_edit.setPlaceholderText("选择左图文件")
        self.right_image_edit.setPlaceholderText("选择右图文件")
        self.sample_name_edit.setClearButtonEnabled(True)
        self.left_image_edit.setClearButtonEnabled(True)
        self.right_image_edit.setClearButtonEnabled(True)
        self.left_image_edit.textChanged.connect(self.refresh_live_preview)
        self.right_image_edit.textChanged.connect(self.refresh_live_preview)
        self._add_form_row(layout, "样本名", self.sample_name_edit)
        self._add_path_row(layout, "左图路径", self.left_image_edit, self.select_left_image)
        self._add_path_row(layout, "右图路径", self.right_image_edit, self.select_right_image)
        self._add_form_row(layout, "内参 / 基线", self.intrinsic_edit)
        layout.addWidget(self.copy_input_checkbox)
        return card

    def _build_model_card(self) -> QWidget:
        card, layout = self._card_with_layout("03 模型与服务", "切换模型时只保留相关接口说明，减少干扰。")
        self.model_combo = QComboBox()
        self.model_combo.addItem("Foundation Stereo", "fs")
        self.model_combo.addItem("S2M2 Stereo Matching", "s2m2")
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.fs_url_edit = QLineEdit()
        self.s2m2_url_edit = QLineEdit()
        self.fs_url_edit.setPlaceholderText("http://127.0.0.1:8000")
        self.s2m2_url_edit.setPlaceholderText("http://127.0.0.1:8001")
        self.fs_url_edit.setClearButtonEnabled(True)
        self.s2m2_url_edit.setClearButtonEnabled(True)
        self.service_status_label = QLabel("尚未检查服务")
        self.service_status_label.setObjectName("StatusNote")
        self.service_status_label.setWordWrap(True)
        self._add_form_row(layout, "当前模型", self.model_combo)
        self._add_form_row(layout, "FS 服务地址", self.fs_url_edit)
        self._add_form_row(layout, "S2M2 服务地址", self.s2m2_url_edit)
        layout.addWidget(self.service_status_label)
        return card

    def _build_params_card(self) -> QWidget:
        card, layout = self._card_with_layout("04 参数配置", "只展示当前模型需要的参数，避免整个左栏看起来像一张超长表单。")
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
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(1, 1)
        for key in ["scale", "iters", "z_far", "d_nb", "d_rad"]:
            self.fs_param_widgets[key] = QLineEdit()
            self.fs_param_widgets[key].setAlignment(Qt.AlignRight)
        self.fs_param_widgets["hiera"] = QCheckBox("开启分层推理")
        self.fs_param_widgets["get_pc"] = QCheckBox("生成点云 .ply")
        self.fs_param_widgets["rm_inv"] = QCheckBox("移除不可见点")
        self.fs_param_widgets["denoise"] = QCheckBox("点云去噪")
        rows = [("缩放比例", "scale"), ("迭代次数", "iters"), ("Z-Far", "z_far"), ("去噪邻域点数", "d_nb"), ("去噪半径", "d_rad")]
        for row, (label_text, key) in enumerate(rows):
            label = QLabel(label_text)
            label.setObjectName("InlineLabel")
            self.fs_param_widgets[key].setPlaceholderText({"scale": "1.0", "iters": "32", "z_far": "10.0", "d_nb": "30", "d_rad": "0.03"}[key])
            layout.addWidget(label, row, 0)
            layout.addWidget(self.fs_param_widgets[key], row, 1)
        row = len(rows)
        for offset, key in enumerate(["hiera", "get_pc", "rm_inv", "denoise"]):
            layout.addWidget(self.fs_param_widgets[key], row + offset, 0, 1, 2)
        return page

    def _build_s2m2_params_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(1, 1)

        model_choice = QComboBox()
        for item in ["Small (S)", "Medium (M)", "Large (L)", "X-Large (XL)"]:
            model_choice.addItem(item)

        self.s2m2_param_widgets["model_choice"] = model_choice
        for key in ["z_far", "conf_thresh", "d_nb", "d_rad"]:
            self.s2m2_param_widgets[key] = QLineEdit()
            self.s2m2_param_widgets[key].setAlignment(Qt.AlignRight)
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
            label = QLabel(label_text)
            label.setObjectName("InlineLabel")
            if key != "model_choice":
                self.s2m2_param_widgets[key].setPlaceholderText({"z_far": "10.0", "conf_thresh": "0.0", "d_nb": "30", "d_rad": "0.03"}[key])
            layout.addWidget(label, row, 0)
            layout.addWidget(self.s2m2_param_widgets[key], row, 1)
        row = len(rows)
        layout.addWidget(self.s2m2_param_widgets["get_pc"], row, 0, 1, 2)
        layout.addWidget(self.s2m2_param_widgets["denoise"], row + 1, 0, 1, 2)
        return page

    def _build_action_card(self) -> QWidget:
        card, layout = self._card_with_layout("05 执行控制", "主操作只保留开始与停止，其他动作退居二级，整体阅读会轻很多。")
        summary_row = QHBoxLayout()
        summary_row.setSpacing(8)
        self.action_model_chip = QLabel("当前模型 -")
        self.action_model_chip.setObjectName("SoftPill")
        self.output_dir_chip = QLabel("输出目录 -")
        self.output_dir_chip.setObjectName("SoftPill")
        summary_row.addWidget(self.action_model_chip)
        summary_row.addWidget(self.output_dir_chip)
        summary_row.addStretch(1)

        utility_row = QHBoxLayout()
        utility_row.setSpacing(10)
        run_row = QHBoxLayout()
        run_row.setSpacing(10)

        self.save_button = QPushButton("保存配置")
        self.check_button = QPushButton("检查服务")
        self.run_button = QPushButton("开始处理")
        self.stop_button = QPushButton("停止任务")
        self.open_output_button = QPushButton("打开输出目录")
        self.save_button.setProperty("variant", "secondary")
        self.check_button.setProperty("variant", "secondary")
        self.open_output_button.setProperty("variant", "ghost")
        self.run_button.setProperty("variant", "primary")
        self.stop_button.setProperty("variant", "danger")
        self.run_button.setMinimumHeight(42)
        self.stop_button.setMinimumHeight(42)
        self.stop_button.setMinimumWidth(116)

        self.save_button.clicked.connect(self.save_current_config)
        self.check_button.clicked.connect(self.check_current_service)
        self.run_button.clicked.connect(self.start_processing)
        self.stop_button.clicked.connect(self.stop_processing)
        self.open_output_button.clicked.connect(self.open_output_dir)
        self.stop_button.setEnabled(False)

        utility_row.addWidget(self.save_button)
        utility_row.addWidget(self.check_button)
        utility_row.addStretch(1)
        utility_row.addWidget(self.open_output_button)

        run_row.addWidget(self.run_button, stretch=1)
        run_row.addWidget(self.stop_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_hint = QLabel("等待开始")
        self.progress_hint.setObjectName("ProgressHint")

        layout.addLayout(summary_row)
        layout.addLayout(utility_row)
        layout.addLayout(run_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_hint)
        return card

    def _build_center_panel(self) -> QWidget:
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)

        preview_card = QFrame()
        preview_card.setObjectName("Card")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(18, 18, 18, 18)
        preview_layout.setSpacing(14)
        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        header_text = QVBoxLayout()
        header_text.setSpacing(4)
        title = QLabel("预览工作台")
        title.setObjectName("CardTitle")
        subtitle = QLabel("主画布一次只展示一个关键预览，切换标签即可查看左图、右图、视差结果与辅助产物。")
        subtitle.setObjectName("HintText")
        subtitle.setWordWrap(True)
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        compact_hint = QLabel("单画布优先")
        compact_hint.setObjectName("SoftPill")
        header_row.addLayout(header_text, stretch=1)
        header_row.addWidget(compact_hint, alignment=Qt.AlignTop)
        preview_layout.addLayout(header_row)

        self.left_preview = PreviewPane("左图")
        self.right_preview = PreviewPane("右图")
        self.result_preview = PreviewPane("视差可视化")
        self.aux_preview = PreviewPane("辅助结果")
        self.preview_tabs = QTabWidget()
        self.preview_tabs.setDocumentMode(True)
        self.preview_tabs.addTab(self.left_preview, "左图")
        self.preview_tabs.addTab(self.right_preview, "右图")
        self.preview_tabs.addTab(self.result_preview, "视差结果")
        self.preview_tabs.addTab(self.aux_preview, "辅助结果")
        preview_layout.addWidget(self.preview_tabs, stretch=1)

        log_card = QFrame()
        log_card.setObjectName("Card")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(18, 18, 18, 18)
        log_layout.setSpacing(12)
        log_title = QLabel("运行日志")
        log_title.setObjectName("CardTitle")
        log_subtitle = QLabel("保留服务连接、推理进度和结果写入信息，排查问题时更顺手。")
        log_subtitle.setObjectName("HintText")
        log_subtitle.setWordWrap(True)
        self.log_output = QTextEdit()
        self.log_output.setObjectName("LogOutput")
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QTextEdit.NoWrap)
        self.log_output.setMinimumHeight(220)
        log_layout.addWidget(log_title)
        log_layout.addWidget(log_subtitle)
        log_layout.addWidget(self.log_output, stretch=1)

        splitter.addWidget(preview_card)
        splitter.addWidget(log_card)
        splitter.setSizes([670, 250])
        return splitter

    def _build_side_panel(self) -> QWidget:
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([580, 360])
        return splitter

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)

        history_card, history_layout = self._card_with_layout("结果记录", "按最近处理时间倒序展示，选中后会联动预览、文件和摘要。")
        self.results_list = QListWidget()
        self.results_list.setSpacing(6)
        self.results_list.itemSelectionChanged.connect(self.on_result_selected)
        history_layout.addWidget(self.results_list)

        file_card, file_layout = self._card_with_layout("输出文件", "双击条目即可打开所在目录，适合快速定位产物。")
        self.output_files_list = QListWidget()
        self.output_files_list.setSpacing(6)
        self.output_files_list.itemDoubleClicked.connect(self.open_selected_artifact)
        file_layout.addWidget(self.output_files_list)

        meta_card, meta_layout = self._card_with_layout("结果摘要", "记录模型、耗时、参数与路径，便于回查。")
        self.metadata_title_label = QLabel("当前未选中结果")
        self.metadata_title_label.setObjectName("InlineLabel")
        self.metadata_view = QTextEdit()
        self.metadata_view.setObjectName("JsonViewer")
        self.metadata_view.setReadOnly(True)
        self.metadata_view.setLineWrapMode(QTextEdit.NoWrap)
        self.metadata_view.setMinimumHeight(220)
        meta_layout.addWidget(self.metadata_title_label)
        meta_layout.addWidget(self.metadata_view)

        splitter.addWidget(history_card)
        splitter.addWidget(file_card)
        splitter.addWidget(meta_card)
        splitter.setSizes([290, 230, 300])
        layout.addWidget(splitter, stretch=1)
        return container

    def _card_with_layout(self, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
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
        button.setProperty("variant", "secondary")
        button.setProperty("small", True)
        button.setMinimumWidth(76)
        button.clicked.connect(browse_handler)
        row.addWidget(line_edit, stretch=1)
        row.addWidget(button)
        layout.addWidget(label)
        layout.addLayout(row)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #edf2f6;
            }
            QWidget {
                color: #1d2733;
            }
            #ToolbarCard {
                background: #fffdfa;
                border: 1px solid #d8e0e7;
                border-radius: 24px;
            }
            #ToolbarEyebrow {
                color: #2d6972;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            #ToolbarTitle {
                color: #1d2d3a;
                font-size: 26px;
                font-weight: 700;
            }
            #ToolbarSubtitle {
                color: #607283;
                font-size: 13px;
                line-height: 1.45;
            }
            #MetricPill, #SoftPill {
                background: #eef4f6;
                color: #3e5565;
                border: 1px solid #d8e2e8;
                border-radius: 12px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: 700;
            }
            #StatusLabel {
                color: #607283;
                font-size: 11px;
                font-weight: 700;
            }
            #StatusChip {
                background: #e8eff2;
                color: #274750;
                border: 1px solid #cedae0;
                border-radius: 14px;
                padding: 7px 12px;
                font-weight: 700;
            }
            #StatusChip[state="running"] {
                background: #fff1dd;
                border-color: #f2d7ab;
                color: #89561b;
            }
            #StatusChip[state="success"] {
                background: #e7f6ee;
                border-color: #b9ddc8;
                color: #1f6641;
            }
            #StatusChip[state="error"] {
                background: #fde8e3;
                border-color: #efc0b4;
                color: #8b412c;
            }
            #StatusValue {
                color: #435767;
                font-size: 12px;
                line-height: 1.45;
            }
            #Card {
                background: #fffdfa;
                border: 1px solid #d8e0e7;
                border-radius: 20px;
            }
            #CardTitle { font-size: 17px; font-weight: 700; color: #203142; }
            #FieldLabel {
                font-size: 12px;
                font-weight: 700;
                color: #4e6273;
                margin-top: 2px;
            }
            #InlineLabel {
                font-size: 12px;
                font-weight: 700;
                color: #4e6273;
            }
            #HintText, #PathCaption, #StatusNote, #ProgressHint {
                font-size: 12px;
                color: #66798a;
            }
            #PathCaption { font-size: 11px; }
            #PreviewSurface {
                background: #f7fafc;
                border: 1px dashed #ccd7e0;
                border-radius: 16px;
                color: #627487;
                padding: 18px;
            }
            QLineEdit, QTextEdit, QComboBox, QListWidget {
                background: #ffffff;
                border: 1px solid #d5dee6;
                border-radius: 12px;
                padding: 10px 12px;
                selection-background-color: #1f7a6a;
                selection-color: white;
            }
            QTextEdit { padding: 12px; }
            QLineEdit:hover, QTextEdit:hover, QComboBox:hover, QListWidget:hover { border-color: #adc1cf; }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QListWidget:focus { border: 1px solid #1f7a6a; }
            QTextEdit#LogOutput, QTextEdit#JsonViewer {
                background: #f6f9fb;
                border: 1px solid #d7e0e7;
                font-family: "Consolas";
                font-size: 10pt;
            }
            QTabWidget::pane {
                background: #fffdfa;
                border: 1px solid #d8e0e7;
                border-radius: 16px;
                top: -1px;
            }
            QTabBar::tab {
                background: #edf3f6;
                color: #4f6474;
                border: 1px solid #d8e0e7;
                border-bottom-color: #d8e0e7;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 8px 14px;
                margin-right: 6px;
                font-weight: 700;
            }
            QTabBar::tab:selected {
                background: #fffdfa;
                color: #203142;
                border-bottom-color: #fffdfa;
            }
            QTabBar::tab:hover:!selected {
                background: #e4edf2;
            }
            QPushButton {
                background: #e9eef1;
                border: 1px solid transparent;
                color: #314252;
                border-radius: 12px;
                padding: 10px 16px;
                font-weight: 700;
            }
            QPushButton[variant="primary"] {
                background: #1f7a6a;
                border-color: #1f7a6a;
                color: white;
            }
            QPushButton[variant="primary"]:hover {
                background: #186558;
                border-color: #186558;
            }
            QPushButton[variant="secondary"] {
                background: #eef4f6;
                border-color: #d2dee4;
                color: #2d4954;
            }
            QPushButton[variant="secondary"]:hover {
                background: #e3edf1;
                border-color: #c1d2da;
            }
            QPushButton[variant="ghost"] {
                background: #fffdfa;
                border-color: #d5dee6;
                color: #445464;
            }
            QPushButton[variant="ghost"]:hover {
                background: #f4f8fa;
                border-color: #c5d5de;
            }
            QPushButton[variant="danger"] {
                background: #b85e3f;
                border-color: #b85e3f;
                color: white;
            }
            QPushButton[variant="danger"]:hover {
                background: #a75135;
                border-color: #a75135;
            }
            QPushButton[small="true"] { padding: 9px 14px; border-radius: 11px; }
            QPushButton:disabled {
                background: #cbc2b7;
                border-color: #cbc2b7;
                color: #faf7f2;
            }
            QCheckBox { spacing: 8px; color: #2e3d4b; padding: 2px 0; }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 6px;
                border: 1px solid #b7c5d0;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #1f7a6a;
                border-color: #1f7a6a;
            }
            QProgressBar {
                background: #e5ecf1;
                border: none;
                border-radius: 11px;
                min-height: 18px;
                text-align: center;
            }
            QProgressBar::chunk { background: #2e8d68; border-radius: 11px; }
            QListWidget {
                padding: 8px;
                outline: 0;
            }
            QListWidget::item {
                border-radius: 10px;
                padding: 10px 12px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: #e1eee9;
                color: #163d35;
                border: 1px solid #a9cdbe;
            }
            QListWidget::item:hover:!selected { background: #eff4f7; }
            QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 4px 0 4px 0;
            }
            QScrollBar::handle:vertical {
                background: #c4d0d9;
                border-radius: 5px;
                min-height: 28px;
            }
            QScrollBar::handle:vertical:hover { background: #afbfca; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
                border: none;
            }
            QSplitter::handle { background: transparent; }
            QSplitter::handle:hover {
                background: rgba(66, 95, 112, 0.08);
                border-radius: 4px;
            }
            """
        )

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _update_responsive_layout(self, force: bool = False) -> None:
        if not hasattr(self, "main_splitter"):
            return
        compact_window = self.width() < 1360
        if not force and compact_window == self._compact_window:
            return
        self._compact_window = compact_window
        orientation = Qt.Vertical if compact_window else Qt.Horizontal
        self.main_splitter.setOrientation(orientation)
        if compact_window:
            self.main_splitter.setSizes([int(self.height() * 0.56), int(self.height() * 0.44)])
            return
        self.main_splitter.setSizes([int(self.width() * 0.62), int(self.width() * 0.38)])

    def _set_status(self, state: str, chip_text: str, detail: str) -> None:
        state_map = {
            "ready": "ready",
            "running": "running",
            "success": "success",
            "error": "error",
        }
        self.status_chip.setText(chip_text)
        self.status_chip.setProperty("state", state_map.get(state, "ready"))
        self.status_chip.style().unpolish(self.status_chip)
        self.status_chip.style().polish(self.status_chip)
        self.status_chip.update()
        self.status_value.setText(detail)

    def _sync_summary_chips(self) -> None:
        if not hasattr(self, "model_combo"):
            return
        model_text = self.model_combo.currentText().replace(" Stereo Matching", "")
        output_dir = self.output_dir_edit.text().strip() or self.config["paths"]["output_dir"]
        output_name = Path(output_dir).name or output_dir
        record_count = len(self.result_records)
        self.model_summary_chip.setText(f"模型 {model_text}")
        self.result_count_chip.setText(f"记录 {record_count}")
        self.action_model_chip.setText(f"当前模型 {self.model_combo.currentText()}")
        self.output_dir_chip.setText(f"输出目录 {output_name}")

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
        self._sync_summary_chips()

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
        self._sync_summary_chips()

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
        self._set_status("running", "运行中", "正在调用模型服务")
        self.worker.start()

    def stop_processing(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.append_log("已请求停止，等待当前网络请求结束...")
            self.stop_button.setEnabled(False)
            self._set_status("running", "停止中", "等待当前请求安全结束")

    def on_processing_finished(self, success: bool, message: str, metadata) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._set_status("success" if success else "error", "完成" if success else "失败", message)
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
        self._set_status("running", "运行中", f"{text} · {sample_name}")
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
        self._sync_summary_chips()

        selected_item = None
        for record in self.result_records:
            label = f"{record.get('sample_name', 'unknown')} · {record.get('model', '').upper()}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, record)
            self.results_list.addItem(item)
            if select_sample and record.get("sample_name") == select_sample:
                selected_item = item

        if self.results_list.count() == 0:
            self.metadata_title_label.setText("当前未选中结果")
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
        self.metadata_title_label.setText(f"当前记录：{record.get('sample_name', 'unknown')} · {record.get('model', '').upper()}")

        left_path = inputs.get("left_image", "")
        right_path = inputs.get("right_image", "")
        if left_path and Path(left_path).exists():
            self.left_preview.set_image(left_path, left_path)
        else:
            self.left_preview.set_empty("缺少左图", "当前记录没有可显示的左图输入。")
        if right_path and Path(right_path).exists():
            self.right_preview.set_image(right_path, right_path)
        else:
            self.right_preview.set_empty("缺少右图", "当前记录没有可显示的右图输入。")

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
            self._set_status("ready", "就绪", "服务检查成功")
        except Exception as exc:  # noqa: BLE001
            self.service_status_label.setText(f"服务检查失败：{exc}")
            self.append_log(f"服务检查失败：{server_url} -> {exc}")
            self._set_status("error", "异常", "服务检查失败")

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
            self._sync_summary_chips()
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
