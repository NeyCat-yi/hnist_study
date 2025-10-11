# QT_bird_test.py
from PySide6 import QtWidgets, QtCore, QtGui
import cv2, os, sys, time, platform
from threading import Thread
from pathlib import Path
import sys, shutil, glob
import re

os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO


# =========================
# 主题（小清新）
# =========================
class ThemeManager:
    THEMES = {
        "薄荷": {
            "accent": "#7CD6CF",
            "accent_hover": "#69c8c0",
            "accent_press": "#58b6ae",
            "bg": "#F7FAFC",
            "card": "#FFFFFF",
            "border": "#D7E2F9",
            "text": "#1F2937",
            "muted": "#6B7280",
            "header_grad_a": "#ffffffff",
            "header_grad_b": "#f6fbf9ff",
        },
        "天空": {
            "accent": "#8BD3DD",
            "accent_hover": "#79c5cf",
            "accent_press": "#68b3bd",
            "bg": "#F5F7FB",
            "card": "#FFFFFF",
            "border": "#DDE7FF",
            "text": "#1E293B",
            "muted": "#667085",
            "header_grad_a": "#ffffffff",
            "header_grad_b": "#eef5ffff",
        },
    }

    @staticmethod
    def qss(p):
        return f"""
        QWidget {{
            background: {p['bg']};
            color: {p['text']};
            font-family: "SF Pro Text","PingFang SC","Microsoft YaHei","Segoe UI",Arial;
            font-size: 13px;
        }}
        #RootCard {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 16px;
        }}
        QLabel#videoCard {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 14px;
        }}
        QGroupBox {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 12px;
            margin-top: 10px; padding: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; left: 12px; padding: 2px 6px; color: {p['muted']};
        }}
        QTextBrowser {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 10px; padding: 8px;
        }}
        QPushButton {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 10px; padding: 8px 12px;
        }}
        QPushButton:hover {{ border-color: {p['accent_hover']}; }}
        QPushButton:pressed {{ border-color: {p['accent_press']}; background: #f5f7f9; }}
        QPushButton#accent {{
            background: {p['accent']}; color: #073b3a; border: none;
        }}
        QPushButton#accent:hover {{ background: {p['accent_hover']}; }}
        QPushButton#accent:pressed {{ background: {p['accent_press']}; }}
        QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {p['card']}; border: 1px solid {p['border']};
            border-radius: 8px; padding: 6px 10px;
        }}
        QSlider::groove:horizontal {{ height: 6px; background: {p['border']}; border-radius: 3px; }}
        QSlider::handle:horizontal {{ background: {p['accent']}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
        QToolButton#styleBtn {{
            border: none; background: transparent; padding: 6px 10px; border-radius: 8px; color: {p['muted']};
        }}
        QToolButton#styleBtn:hover {{
            background: #00000010;
        }}
        """

    @staticmethod
    def apply(win, theme="薄荷", custom_accent=None):
        pal = dict(ThemeManager.THEMES[theme])
        if custom_accent:
            pal["accent"] = custom_accent
        win._palette_cache = pal
        win.setStyleSheet(ThemeManager.qss(pal))


# =========================
# 推理后台（信号回主线程）
# =========================
class FrameProcessor(QtCore.QObject):
    processed = QtCore.Signal(QtGui.QImage)
    original  = QtCore.Signal(QtGui.QImage)
    status    = QtCore.Signal(str)
    fps_sig   = QtCore.Signal(float)

    def __init__(self):
        super().__init__()
        self.model = None
        self.frame_queue = []
        self.running = True
        self.enable_detect = True
        self.conf = 0.25
        self._last = time.time()
        self._cnt = 0

    def load_model(self, path, device="cpu"):
        try:
            self.model = YOLO(path)
            if device.lower() != "cpu":
                try:
                    self.model.to(device)
                except Exception:
                    self.status.emit("⚠️ GPU 切换失败，已回退 CPU。")
            self.status.emit(f"✅ 使用设备： @ {device}")
        except Exception as e:
            self.status.emit(f"❌ 模型加载失败：{e}")

    def push(self, frame_rgb_520x400):
        self.frame_queue = [frame_rgb_520x400]  # 仅保留最新帧

    def loop(self):
        while self.running:
            if not self.model:
                time.sleep(0.02); continue
            if not self.frame_queue:
                time.sleep(0.01); continue

            frame = self.frame_queue.pop(0)
            self.original.emit(QtGui.QImage(
                frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888))

            try:
                if self.enable_detect:
                    results = self.model(frame, conf=self.conf)[0]
                    img = results.plot(line_width=1)
                else:
                    img = frame
                self.processed.emit(QtGui.QImage(
                    img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888))
            except Exception as e:
                self.status.emit(f"❌ 推理失败：{e}")

            self._cnt += 1
            now = time.time()
            if now - self._last >= 1.0:
                self.fps_sig.emit(self._cnt / (now - self._last))
                self._cnt = 0
                self._last = now
            time.sleep(0.01)


# =========================
# macOS 风格 交通灯按钮
# =========================
class MacWindowButton(QtWidgets.QToolButton):
    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind  # 'close' | 'min' | 'zoom'
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(16, 16)
        self.setStyleSheet("border:none; background:transparent;")
        self._hover = False

    def enterEvent(self, e): self._hover = True;  self.update()
    def leaveEvent(self, e): self._hover = False; self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        colors = {
            'close': QtGui.QColor("#FF5F57"),
            'min':   QtGui.QColor("#FFBD2E"),
            'zoom':  QtGui.QColor("#28C840"),
        }
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(colors[self.kind])
        p.drawEllipse(self.rect())
        if self._hover:
            pen = QtGui.QPen(QtGui.QColor(0,0,0,120), 1.6)
            p.setPen(pen)
            cx, cy = self.width()/2, self.height()/2
            if self.kind == 'close':
                p.drawLine(cx-3, cy-3, cx+3, cy+3)
                p.drawLine(cx-3, cy+3, cx+3, cy-3)
            elif self.kind == 'min':
                p.drawLine(cx-3.5, cy, cx+3.5, cy)
            elif self.kind == 'zoom':
                p.drawLine(cx-3, cy, cx+3, cy)
                p.drawLine(cx, cy-3, cx, cy+3)


# =========================
# 自定义 macOS 标题栏
# =========================
class MacTitleBar(QtWidgets.QWidget):
    request_min   = QtCore.Signal()
    request_zoom  = QtCore.Signal()
    request_close = QtCore.Signal()

    theme_selected = QtCore.Signal(str)   # "薄荷"/"天空"
    request_accent = QtCore.Signal()      # 触发选色器
    request_reset  = QtCore.Signal()      # 恢复默认

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(8)

        btnWrap = QtWidgets.QWidget()
        btnLay = QtWidgets.QHBoxLayout(btnWrap)
        btnLay.setContentsMargins(0, 0, 0, 0); btnLay.setSpacing(8)
        self.btnClose = MacWindowButton('close')
        self.btnMin   = MacWindowButton('min')
        self.btnZoom  = MacWindowButton('zoom')
        btnLay.addWidget(self.btnClose); btnLay.addWidget(self.btnMin); btnLay.addWidget(self.btnZoom)

        self.title = QtWidgets.QLabel("Mineral Flotation")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 14px; font-weight: 600; color: #3a3a3a;")

        rightWrap = QtWidgets.QWidget()
        rLay = QtWidgets.QHBoxLayout(rightWrap)
        rLay.setContentsMargins(0,0,0,0); rLay.setSpacing(8)

        self.styleBtn = QtWidgets.QToolButton(objectName="styleBtn")
        self.styleBtn.setText("样式")
        self.styleBtn.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        menu = QtWidgets.QMenu(self.styleBtn)
        sub_theme = menu.addMenu("主题")
        act_mint  = sub_theme.addAction("薄荷")
        act_sky   = sub_theme.addAction("天空")
        menu.addSeparator()
        act_accent = menu.addAction("选择点缀色…")
        act_reset  = menu.addAction("恢复默认")
        self.styleBtn.setMenu(menu)

        act_mint.triggered.connect(lambda: self.theme_selected.emit("薄荷"))
        act_sky.triggered.connect(lambda: self.theme_selected.emit("天空"))
        act_accent.triggered.connect(self.request_accent.emit)
        act_reset.triggered.connect(self.request_reset.emit)

        rLay.addWidget(self.styleBtn)

        lay.addWidget(btnWrap, 0, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        lay.addWidget(self.title, 1)
        lay.addWidget(rightWrap, 0, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)

        self.btnClose.clicked.connect(self.request_close.emit)
        self.btnMin.clicked.connect(self.request_min.emit)
        self.btnZoom.clicked.connect(self.request_zoom.emit)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pal = self.parent()._palette_cache if hasattr(self.parent(), "_palette_cache") else ThemeManager.THEMES["薄荷"]
        grad = QtGui.QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QtGui.QColor(pal["header_grad_a"]))
        grad.setColorAt(1.0, QtGui.QColor(pal["header_grad_b"]))
        p.fillRect(self.rect(), QtGui.QBrush(grad))

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self._drag_pos = e.globalPosition().toPoint()
            self._win_pos = self.window().frameGeometry().topLeft()
    def mouseMoveEvent(self, e):
        if e.buttons() & QtCore.Qt.LeftButton:
            delta = e.globalPosition().toPoint() - self._drag_pos
            self.window().move(self._win_pos + delta)
    def mouseDoubleClickEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            self.request_zoom.emit()


# =========================
# 主窗口
# =========================
class MWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Window)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.resize(1280, 860)

        self._imgLogDlg = None  # 图片处理日志弹窗（非必用，保留）
        # —— 新增：图片日志缓存（仅供解析表格）
        self._img_stdout_buffer = []

        # 新增—— 类别与备注映射
        self.supList = ["严重欠浮选", "轻微欠浮选", "正常", "轻微过浮选", "严重过浮选"]
        self.class_notes = {
            "1": "视觉特征：大尺寸，浅色，粗糙质地，表面光滑。\n解决办法：大幅增捕收剂/充气，延时",
            "2": "视觉特征：相对较大尺寸，颜色较浅，质地较粗。\n解决办法：微增捕收剂，适度加气",
            "3": "视觉特征：中等大小，分布均匀，整体外观稳定。\n解决办法：维持药剂/气量，稳控pH与液位",
            "4": "视觉特征：体积小，有积聚，颜色较深。\n解决办法：减起泡剂，增泡层深度",
            "5": "视觉特征：更小的尺寸，更细腻的质地，带有许多皱褶和泥泞的区域。\n解决办法：降捕收剂/充气，强抑杂质并洗涤",
        }
        self._last_pred_class = None

        # 预览播放器
        self.previewCap = None
        self.previewTimer = QtCore.QTimer(self)
        self.previewTimer.timeout.connect(self._preview_loop)

        # 外部脚本输出缓存（视频）
        self._infer_stdout_buffer = []
        self._last_infer_text = ""
        self._last_video_path = ""
        self._infer_start_ts = None

        # 根卡片 + 阴影
        root = QtWidgets.QWidget(objectName="RootCard")
        shadow = QtWidgets.QGraphicsDropShadowEffect(blurRadius=30, xOffset=0, yOffset=18)
        shadow.setColor(QtGui.QColor(0, 0, 0, 40))
        root.setGraphicsEffect(shadow)

        rootLay = QtWidgets.QVBoxLayout(root)
        rootLay.setContentsMargins(0, 0, 0, 12)
        rootLay.setSpacing(0)

        # 标题栏
        self.titleBar = MacTitleBar(self)
        rootLay.addWidget(self.titleBar)

        # 中心区
        central = QtWidgets.QWidget()
        rootLay.addWidget(central)
        main = QtWidgets.QVBoxLayout(central)
        main.setContentsMargins(16, 10, 16, 0)
        main.setSpacing(12)

        # 上：视频双卡片
        top = QtWidgets.QHBoxLayout(); top.setSpacing(12)
        self.label_ori_video = QtWidgets.QLabel(objectName="videoCard")
        self.label_treated   = QtWidgets.QLabel(objectName="videoCard")
        for lab, tip in [(self.label_ori_video,"Original"),(self.label_treated,"Result")]:
            lab.setMinimumSize(580, 420)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setToolTip(tip)
        top.addWidget(self._wrap_caption(self.label_ori_video, "Original"))
        top.addWidget(self._wrap_caption(self.label_treated, "Result"))
        main.addLayout(top)

        # 中：控制条
        ctrl = QtWidgets.QHBoxLayout(); ctrl.setSpacing(10)
        self.btnOpen = QtWidgets.QPushButton("🎞️ 浮选工况分类", objectName="accent")

        # —— 查看分类结果（视频）
        self.btnShowResult = QtWidgets.QPushButton("👁️ 浮选分类结果")
        self.btnShowResult.setEnabled(False)
        self.btnShowResult.setToolTip("推理结束后可点击预览")
        self.btnShowResult.clicked.connect(self.show_infer_result_dialog)
        ctrl.addWidget(self.btnShowResult)

        self.btnCam  = QtWidgets.QPushButton("📹 摄像头", objectName="accent")
        self.btnStop = QtWidgets.QPushButton("🛑 停止")
        self.toggleDetect = QtWidgets.QPushButton("开启检测"); self.toggleDetect.setCheckable(True); self.toggleDetect.setChecked(True)

        self.confLabel = QtWidgets.QLabel("置信度: 0.25")
        self.confSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.confSlider.setRange(10,90); self.confSlider.setValue(25)

        self.modelBox = QtWidgets.QComboBox(); self.modelBox.addItems(["yolov8n.pt","yolov8s.pt"])
        self.deviceBox = QtWidgets.QComboBox(); self.deviceBox.addItems(["cpu","cuda:0"])
        self.btnSnap = QtWidgets.QPushButton("📸 截图")
        self.btnSave = QtWidgets.QPushButton("💾 保存帧")

        # —— 打开图片并调用外部脚本
        self.btnOpenImage = QtWidgets.QPushButton("🖼️ 静态特征提取")
        ctrl.addWidget(self.btnOpenImage) 

        # —— 新增：查看“特征表”按钮
        self.btnShowImageFeatures = QtWidgets.QPushButton("📊 查看特征表")
        self.btnShowImageFeatures.setEnabled(False)
        self.btnShowImageFeatures.setToolTip("图片处理完成后查看 17 项形态学特征")
        self.btnShowImageFeatures.clicked.connect(self.show_image_features_dialog)
        ctrl.addWidget(self.btnShowImageFeatures)

        ctrl.addWidget(self.btnOpen); ctrl.addWidget(self.btnCam); ctrl.addWidget(self.btnStop)
        ctrl.addSpacing(8)
        #ctrl.addWidget(self.confLabel); ctrl.addWidget(self.confSlider,1) # 暂时不需要置信度修改
        ctrl.addSpacing(8)
        ctrl.addWidget(QtWidgets.QLabel("设备"));  ctrl.addWidget(self.deviceBox)
        ctrl.addWidget(self.btnSnap); ctrl.addWidget(self.btnSave)
        main.addLayout(ctrl)

        # 进度/时间
        self.progress = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.progress.setRange(0,0)
        self.timeLabel = QtWidgets.QLabel("--:-- / --:--")
        main.addWidget(self.progress)
        main.addWidget(self.timeLabel, alignment=QtCore.Qt.AlignRight)

        # 下：日志
        group = QtWidgets.QGroupBox("运行日志")
        gLay = QtWidgets.QHBoxLayout(group)
        self.textLog = QtWidgets.QTextBrowser()
        gLay.addWidget(self.textLog)
        main.addWidget(group)

        # 外层布局
        wrap = QtWidgets.QVBoxLayout(self)
        wrap.setContentsMargins(18, 18, 18, 18)
        wrap.addWidget(root)

        # 标题栏信号
        self.titleBar.request_close.connect(self.close)
        self.titleBar.request_min.connect(self.showMinimized)
        self.titleBar.request_zoom.connect(self.toggleZoom)

        # 样式按钮（下拉菜单）信号
        self.titleBar.theme_selected.connect(self.apply_theme_from_titlebar)
        self.titleBar.request_accent.connect(self.pick_accent_from_titlebar)
        self.titleBar.request_reset.connect(lambda: ThemeManager.apply(self, "薄荷"))

        # 控件信号
        self.btnOpenImage.clicked.connect(self.open_image_and_run)
        self.btnOpen.clicked.connect(self.open_video_and_infer)
        self.btnCam.clicked.connect(self.start_camera)
        self.btnStop.clicked.connect(self.stop)
        self.toggleDetect.toggled.connect(self.on_toggle_detect)
        self.confSlider.valueChanged.connect(self.on_conf_change)
        self.btnSnap.clicked.connect(self.snapshot)
        self.btnSave.clicked.connect(self.save_current_frame)
        self.progress.sliderReleased.connect(self.seek_video)

        # 定时器
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.play_loop)

        # 推理后台
        self.processor = FrameProcessor()
        self.thread = Thread(target=self.processor.loop, daemon=True)
        self.processor.processed.connect(self.update_treated)
        self.processor.original.connect(self.update_original)
        self.processor.status.connect(self.log)
        self.processor.fps_sig.connect(self.on_fps)
        self.thread.start()

        # 默认主题与模型
        ThemeManager.apply(self, "薄荷")
        self.load_model()

        # 变量
        self.cap = None
        self.total_frames = 0
        self._last_qimg = None
        self._last_pix = None

    # ---------- UI 辅助 ----------
    def _wrap_caption(self, widget, text):
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(w); l.setContentsMargins(0,0,0,0)
        cap = QtWidgets.QLabel(text); cap.setAlignment(QtCore.Qt.AlignLeft)
        cap.setStyleSheet("color:#64748b; font-size:12.5px;")
        l.addWidget(cap); l.addWidget(widget)
        return w

    def toggleZoom(self):
        if self.isMaximized(): self.showNormal()
        else: self.showMaximized()

    # ---------- 样式（标题栏按钮） ----------
    def apply_theme_from_titlebar(self, theme):
        ThemeManager.apply(self, theme)

    def pick_accent_from_titlebar(self):
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._palette_cache["accent"]), self, "选择点缀色")
        if col.isValid():
            ThemeManager.apply(self, theme=self._current_theme_name(), custom_accent=col.name())

    def _current_theme_name(self):
        return "薄荷" if self._palette_cache.get("border") == ThemeManager.THEMES["薄荷"]["border"] else "天空"

    # ---------- 业务逻辑 ----------
    def load_model(self):
        self.processor.load_model(self.modelBox.currentText(), self.deviceBox.currentText())
        self.modelBox.currentTextChanged.connect(lambda _: self.processor.load_model(self.modelBox.currentText(), self.deviceBox.currentText()))
        self.deviceBox.currentTextChanged.connect(lambda _: self.processor.load_model(self.modelBox.currentText(), self.deviceBox.currentText()))

    def on_toggle_detect(self, checked):
        self.processor.enable_detect = checked
        self.toggleDetect.setText("开启检测" if checked else "关闭检测")

    def on_conf_change(self, v):
        self.processor.conf = v/100.0
        self.confLabel.setText(f"置信度: {self.processor.conf:.2f}")

    def on_fps(self, fps):
        self.titleBar.title.setText(f"YOLO-Qt · macOS 风格 · 小清新 · {fps:.1f} fps")

    # 打开视频/摄像头
    def start_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        if not path: return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "无法打开", "视频文件无法打开。")
            return
        self._attach_cap(cap, f"打开视频：{os.path.basename(path)}")

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "无法打开", "摄像头无法打开。")
            return
        self._attach_cap(cap, "打开摄像头 0")

    def _attach_cap(self, cap, msg):
        self.cap = cap
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.total_frames > 0: self.progress.setRange(0, self.total_frames-1)
        else: self.progress.setRange(0, 0)
        self.log(f"✅ {msg}")
        if not self.timer.isActive(): self.timer.start(30)

    # 播放循环
    def play_loop(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret:
            self.log("⚠️ 视频结束或读取失败。"); self.stop(); return
        frame = cv2.resize(frame, (520, 400))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.processor.push(frame)

        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if self.total_frames > 0:
            self.progress.blockSignals(True)
            self.progress.setValue(max(0, cur-1))
            self.progress.blockSignals(False)
            fpsv = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.timeLabel.setText(f"{self._fmt_time(cur/fpsv)} / {self._fmt_time(self.total_frames/max(fpsv,1))}")
        else:
            self.timeLabel.setText("--:-- / --:--")

    def seek_video(self):
        if not self.cap or self.total_frames <= 0: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.progress.value())

    def _fmt_time(self, secs):
        m = int(secs // 60); s = int(secs % 60)
        return f"{m:02d}:{s:02d}"

    # 预览循环（视频）
    def _preview_loop(self):
        if not self.previewCap:
            self.previewTimer.stop()
            return
        ok, frame = self.previewCap.read()
        if not ok:
            self.previewCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.previewCap.read()
            if not ok:
                return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.label_ori_video.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.label_ori_video.setPixmap(pix)

    # 打开视频并外部推理
    def open_video_and_infer(self):
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频文件用于推理", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv)"
        )
        if not video_path:
            return

        if self.previewTimer.isActive():
            self.previewTimer.stop()
        if self.previewCap:
            self.previewCap.release(); self.previewCap = None

        self.previewCap = cv2.VideoCapture(video_path)
        fps = self.previewCap.get(cv2.CAP_PROP_FPS) or 30
        interval = int(max(10, 1000 / fps))
        self.previewTimer.start(interval)

        if not self.previewCap.isOpened():
            QtWidgets.QMessageBox.warning(self, "无法打开", "视频文件无法打开。")
            return

        ok, frame = self.previewCap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                self.label_ori_video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.label_ori_video.setPixmap(pix)
            self.previewCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.previewTimer.start(30)

        from pathlib import Path; import sys
        script_path = (Path(__file__).resolve().parent / "simple_inference.py")
        if not script_path.exists():
            script_path = Path(r"/mnt/data/simple_inference.py")
        if not script_path.exists():
            QtWidgets.QMessageBox.warning(self, "找不到脚本",
                f"未找到 simple_inference.py：\n{script_path}")
            self.previewTimer.stop()
            if self.previewCap: self.previewCap.release(); self.previewCap=None
            return

        if hasattr(self, "imgProgress"):
            self.imgProgress.setVisible(True); self.imgProgress.setRange(0, 0); self.imgProgress.setFormat("视频推理：正在运行…")
        if hasattr(self, "_set_title_processing"):
            self._set_title_processing(True)

        args = [str(script_path), "--video_path", video_path]
        if getattr(self, "modelWeightPath", ""):
            args += ["--model_path", self.modelWeightPath]
        else:
            self.log("ℹ️ 未选择权重，使用 simple_inference.py 的默认模型。")

        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.setWorkingDirectory(str(script_path.parent))
        self.proc.readyReadStandardOutput.connect(
            lambda: self._append_proc_output(self.proc.readAllStandardOutput()))
        self.proc.finished.connect(self._on_infer_finished)
        self.proc.start(sys.executable, args)

        self.log(f"🚀 已启动外部推理：{script_path.name}")
        self.log(f"   --video_path = {video_path}")
        if getattr(self, "modelWeightPath", ""):
            self.log(f"   --model_path = {self.modelWeightPath}")

        self._last_video_path = video_path
        self._infer_stdout_buffer.clear()
        self._infer_start_ts = time.time()
        self.btnShowResult.setEnabled(False)

    # 打开图片并调用外部脚本
    def open_image_and_run(self):
        img_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not img_path:
            return

        self._img_path = Path(img_path)
        self._current_task = "image"
        self._img_stdout_buffer.clear()          # 清空图片日志缓存
        self.btnShowImageFeatures.setEnabled(False)

        # 【马上在左侧显示原图】
        pix = QtGui.QPixmap(self._img_path.as_posix()).scaled(
            self.label_ori_video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.label_ori_video.setPixmap(pix)

        # 结果重命名
        self._processed_png = self._img_path.parent / "froth_flotation_segmented.png"
        self._seg_dir = self._img_path.parent / (self._img_path.stem + "_seg")
        out_npy = self._img_path.with_suffix("").as_posix() + "_feat.npy"

        script_path = (Path(__file__).resolve().parent / "extract_morphological_features.py")
        if not script_path.exists():
            script_path = Path(r"/mnt/data/extract_morphological_features.py")
        if not script_path.exists():
            QtWidgets.QMessageBox.warning(self, "找不到脚本", f"未找到脚本：\n{script_path}")
            return

        args = [
            str(script_path),
            "--image", self._img_path.as_posix(),
            "--output", out_npy,
            "--save-segmented",
            "--segmented-dir", self._seg_dir.as_posix(),
            "--print-details"
        ]

        self.proc = QtCore.QProcess(self)
        self.proc.setWorkingDirectory(str(script_path.parent))
        self.proc.readyReadStandardOutput.connect(
            lambda: self._append_proc_output(self.proc.readAllStandardOutput()))
        self.proc.readyReadStandardError.connect(
            lambda: self._append_proc_output(self.proc.readAllStandardError()))
        self.proc.finished.connect(self._on_image_proc_finished)
        self.proc.start(sys.executable, args)

        self.log(f"🚀 执行：{script_path.name}")
        self.log(f"   --image = {self._img_path.name}")
        self.log(f"   结果将另存为：{self._processed_png.name}")
        self.log("处理中... 请稍候。")

    # 图片子进程结束
    def _on_image_proc_finished(self, code, status):
        if code != 0:
            self.log(f"❌ 子进程退出 code={code}, status={status}")
            # 即使失败也允许查看到期望表头（表格会显示未找到提示）
            self.btnShowImageFeatures.setEnabled(True)
            return

        # 找分割结果图片
        candidate = None
        if self._seg_dir and self._seg_dir.exists():
            patterns = ["froth_flotation_segmented.png", "*overlay*.png", "*segmented*.png",
                        "*overlay*.jpg", "*segmented*.jpg", "*.png", "*.jpg", "*.jpeg"]
            for pat in patterns:
                files = sorted(self._seg_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
                if files:
                    candidate = files[0]; break

        if candidate is None:
            self.log("⚠️ 未找到分割结果图片，检查脚本的输出命名或参数。")
        else:
            try:
                shutil.copyfile(candidate, self._processed_png)
                self.log(f"✅ 分割结果已保存为：{self._processed_png.as_posix()}")
                pix = QtGui.QPixmap(self._processed_png.as_posix()).scaled(
                    self.label_treated.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.label_treated.setPixmap(pix)
            except Exception as e:
                self.log(f"❌ 保存结果失败：{e}")

        # 启用“查看特征表”按钮
        self.btnShowImageFeatures.setEnabled(True)
        self.log("✅ 图片外部推理完成。可点击“📊 查看特征表”。")

    # 收集子进程输出 + 解析类别
    def _append_proc_output(self, qbytearray):
        try:
            text = bytes(qbytearray).decode("utf-8", "ignore")
        except Exception:
            text = str(qbytearray)
        if not text.strip():
            return

        # 解析分类（命中一次就记住）
        if self._last_pred_class is None:
            keys = "|".join(map(re.escape, self.class_notes.keys()))
            patt = rf"(?:Predicted\s*class|prediction|预测类别|类别|class)\s*[:=>\-：]\s*({keys})"
            m = re.search(patt, text, flags=re.IGNORECASE)
            if m:
                self._last_pred_class = m.group(1)

        for line in text.rstrip().splitlines():
            self.textLog.append(line)
            self._infer_stdout_buffer.append(line)
            if getattr(self, "_current_task", None) == "image":
                self._img_stdout_buffer.append(line)     # ← 关键：图片日志缓存

    # 分类结果表（视频）
    def show_infer_result_dialog(self):
        p = getattr(self, "_palette_cache", {
            "card":"#fff","border":"#dfe6f3","accent":"#7CD6CF","text":"#1f2937","muted":"#667085","bg":"#f7fafc"
        })
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("分类结果与备注")
        dlg.resize(760, 420)
        dlg.setStyleSheet(f"""
        QDialog {{ background: {p['bg']}; }}
        QLabel {{ color:{p['text']}; font-size:13px; }}
        QTableWidget {{
            background:{p['card']}; border:1px solid {p['border']}; border-radius:10px;
            gridline-color:{p['border']};
        }}
        QHeaderView::section {{
            background:{p['card']}; border: none; border-bottom:1px solid {p['border']};
            padding:6px 8px; font-weight:600;
        }}
        QPushButton {{ background:{p['card']}; border:1px solid {p['border']}; border-radius:10px; padding:8px 14px; }}
        QPushButton#accent {{ background:{p['accent']}; color:#073b3a; border:none; }}
        """)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(14,14,14,14); lay.setSpacing(10)

        pred_idx = None
        try:
            pred_idx = int(self._last_pred_class) - 1 if self._last_pred_class else None
        except:
            pred_idx = None

        labPred = QtWidgets.QLabel(f"预测类别：<b>{self.supList[pred_idx] if pred_idx is not None and 0<=pred_idx<len(self.supList) else '（未识别）'}</b>")
        labPred.setStyleSheet("font-size:14px;")
        info = QtWidgets.QHBoxLayout(); info.addWidget(labPred); info.addStretch(1)
        lay.addLayout(info)

        table = QtWidgets.QTableWidget(0, 2, dlg)
        table.setHorizontalHeaderLabels(["类别", "备注"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setWordWrap(True)
        table.setShowGrid(True)

        for k, v in self.class_notes.items():
            r = table.rowCount(); table.insertRow(r)
            itemK = QtWidgets.QTableWidgetItem(self.supList[int(k) - 1])
            itemV = QtWidgets.QTableWidgetItem(v); itemV.setToolTip(v)
            if self._last_pred_class and k == self._last_pred_class:
                itemK.setForeground(QtGui.QBrush(QtGui.QColor("#073b3a")))
                itemV.setForeground(QtGui.QBrush(QtGui.QColor("#073b3a")))
                bg = QtGui.QColor(p["accent"]); bg.setAlpha(60)
                itemK.setBackground(bg); itemV.setBackground(bg)
                f = itemK.font(); f.setBold(True); itemK.setFont(f); itemV.setFont(f)
            table.setItem(r, 0, itemK); table.setItem(r, 1, itemV)
            table.resizeRowToContents(r)

        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        lay.addWidget(table, 1)

        btns = QtWidgets.QHBoxLayout(); btns.addStretch(1)
        btnClose = QtWidgets.QPushButton("关闭")
        btns.addWidget(btnClose)
        lay.addLayout(btns)
        btnClose.clicked.connect(dlg.accept)
        dlg.exec()

    # === 新增：形态学特征表对话框（图片） ===
    def show_image_features_dialog(self):
        """把 _img_stdout_buffer 中的 ‘=== 形态学特征详细信息 ===’ 段落解析成表格显示"""
        text = "\n".join(self._img_stdout_buffer).strip()
        p = getattr(self, "_palette_cache", {
            "card":"#fff","border":"#dfe6f3","accent":"#7CD6CF","text":"#1f2937","muted":"#667085","bg":"#f7fafc"
        })

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("形态学特征")
        dlg.resize(720, 520)
        dlg.setStyleSheet(f"""
        QDialog {{ background: {p['bg']}; }}
        QLabel  {{ color: {p['text']}; font-size:13px; }}
        QTableWidget {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 10px;
            gridline-color: {p['border']};
        }}
        QHeaderView::section {{
            background: {p['card']};
            border: none;
            border-bottom: 1px solid {p['border']};
            padding: 6px 8px;
            font-weight: 600;
        }}
        QPushButton {{
            background: {p['card']};
            border: 1px solid {p['border']};
            border-radius: 10px;
            padding: 8px 14px;
        }}
        QPushButton#accent {{ background: {p['accent']}; color: #073b3a; border: none; }}
        """)

        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(14,14,14,14)
        lay.setSpacing(10)

        title = QtWidgets.QLabel("形态学特征详细信息")
        title.setStyleSheet("font-size:14px; font-weight:600;")
        lay.addWidget(title)

        table = QtWidgets.QTableWidget(0, 3, dlg)
        table.setHorizontalHeaderLabels(["序号", "指标", "数值"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setAlternatingRowColors(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)

        # 解析
        items = []
        if text:
            m_start = re.search(r"^\s*===\s*形态学特征详细信息\s*===\s*$", text, re.M)
            if m_start:
                block = text[m_start.end():]
                for m in re.finditer(r"^\s*(\d+)\.\s*([^:：]+?)\s*[:：]\s*([+-]?\d+(?:\.\d+)?)\s*$", block, re.M):
                    items.append((int(m.group(1)), m.group(2).strip(), float(m.group(3))))
                items.sort(key=lambda x: x[0])

        if items:
            for idx, name, val in items:
                r = table.rowCount(); table.insertRow(r)
                table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(idx)))
                table.setItem(r, 1, QtWidgets.QTableWidgetItem(name))
                itv = QtWidgets.QTableWidgetItem(f"{val:.4f}")
                itv.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                table.setItem(r, 2, itv)
        else:
            r = table.rowCount(); table.insertRow(r)
            table.setItem(r, 0, QtWidgets.QTableWidgetItem("-"))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem("未找到“形态学特征详细信息”"))
            table.setItem(r, 2, QtWidgets.QTableWidgetItem("-"))

        lay.addWidget(table, 1)

        btns = QtWidgets.QHBoxLayout(); btns.addStretch(1)
        btnClose = QtWidgets.QPushButton("关闭")
        btns.addWidget(btnClose)
        lay.addLayout(btns)
        btnClose.clicked.connect(dlg.accept)

        dlg.exec()

    # 打开视频结束后
    def _on_infer_finished(self, code, status):
        if hasattr(self, "imgProgress"): self.imgProgress.setVisible(False)
        if hasattr(self, "_set_title_processing"): self._set_title_processing(False)
        if not self._last_pred_class:
            self._last_pred_class = "（未识别）"

        if self.previewTimer.isActive():
            self.previewTimer.stop()
        if self.previewCap:
            self.previewCap.release(); self.previewCap = None

        exit_code = code if isinstance(code, int) else 0
        if exit_code == 0:
            self.log("✅ 外部推理完成。")
        else:
            self.log(f"❌ 外部推理失败：exit={exit_code}, status={status}")

        end_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        used = f"{int(time.time() - (self._infer_start_ts or time.time()))}s"
        model_path = getattr(self, "modelWeightPath", "") or "(使用脚本默认权重)"
        header = [
            "＝＝ 推理结果报告 ＝＝",
            f"时间：{end_ts}",
            f"耗时：{used}",
            f"视频：{self._last_video_path}",
            f"权重：{model_path}",
            f"退出码：{code}",
            "-"*28,
            "【外部脚本输出】"
        ]
        body = self._infer_stdout_buffer[:] if self._infer_stdout_buffer else ["（无输出）"]
        footer = [
            "-"*28,
            "【自定义内容】",
            "（在此填写你的备注……）"
        ]
        self._last_infer_text = "\n".join(header + body + [""] + footer)
        self.btnShowResult.setEnabled(True)

    # 图像更新
    @QtCore.Slot(QtGui.QImage)
    def update_original(self, qimg):
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.label_ori_video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.label_ori_video.setPixmap(pix)

    @QtCore.Slot(QtGui.QImage)
    def update_treated(self, qimg):
        self._last_qimg = qimg
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.label_treated.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self._last_pix = pix
        self.label_treated.setPixmap(pix)

    # 截图/保存
    def snapshot(self):
        if self.label_ori_video is None:
            self.log("ℹ️ 暂无可截图帧。"); return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存 PNG 截图", "snapshot.png", "PNG 图片 (*.png)")
        if fn: self.label_ori_video.save(fn, "PNG"); self.log(f"📸 已保存：{fn}")

    def save_current_frame(self):
        if self._last_pix is None:
            self.log("ℹ️ 暂无可保存帧。"); return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存处理后帧", "result.jpg", "JPEG 图片 (*.jpg *.jpeg)")
        if fn:
            img = self._last_pix.toImage(); img.save(fn, "JPG"); self.log(f"💾 已保存：{fn}")

    # 停止/关闭
    def stop(self):
        if self.timer.isActive(): self.timer.stop()
        if self.cap: self.cap.release(); self.cap = None
        self.label_ori_video.clear(); self.label_treated.clear()
        self.timeLabel.setText("--:-- / --:--"); self.progress.setRange(0,0)
        self.log("🛑 已停止。")
        if self.previewTimer.isActive():
            self.previewTimer.stop()
        if self.previewCap:
            self.previewCap.release(); self.previewCap = None
        if hasattr(self, "proc") and self.proc and self.proc.state() == QtCore.QProcess.ProcessState.Running:
            self.proc.kill()
            self.log("🛑 已终止外部推理进程。")

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.processor.running = False
        super().closeEvent(e)

    def log(self, msg): self.textLog.append(msg)


# =========================
# 入口
# =========================
if __name__ == "__main__":
    if platform.system() == "Windows":
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    w = MWindow()
    w.show()
    sys.exit(app.exec())
