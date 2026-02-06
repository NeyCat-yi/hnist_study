# 视频流处理 (Video Stream Processing)

## 📁 项目结构
- **MediaServer/**: 流媒体服务器相关文件
- **海康工业相机推流/**: 海康威视工业相机取流并推流到 RTSP 服务器的 Python 脚本

## 🚀 快速开始 (Quick Start)

本指南主要针对 **海康工业相机推流** 模块。

### 1. 环境准备 (Prerequisites)

- **操作系统**: Windows (推荐) / Linux
- **Python**: 3.11+
- **FFmpeg**: 必须安装 FFmpeg 并将其添加到系统环境变量 PATH 中。
  - 验证安装: 在终端输入 `ffmpeg -version`，应能看到版本信息。
- **海康威视 SDK**: 
  - 项目中已包含 Python 封装包 (`MvImport` 文件夹)。
  - 需确保已安装海康相机的驱动程序 (MVS)，以便系统能识别到相机。

### 2. 安装依赖 (Installation)

在 `视频流处理` 根目录下，执行以下命令安装 Python 依赖库：

```bash
pip install -r requirements.txt
```

> 依赖包含:
> - `numpy`: 用于图像数据处理
> - `opencv-python`: 用于图像处理和辅助功能

### 3. 配置 (Configuration)

打开 `海康工业相机推流/海康.py` 文件，根据实际环境修改以下配置项：

```python
# ================= 配置区域 =================
TARGET_IP = "169.254.211.253"            # 你的海康相机 IP 地址
RTSP_URL = "rtsp://127.0.0.1:9994/live/stream"  # RTSP 推流地址
FPS = 25                                 # 推流帧率 (建议与相机采集帧率一致)
USE_NVIDIA_GPU = True                    # 是否开启 NVIDIA 显卡硬件加速 (True/False)
# ===========================================
```

### 4. 运行 (Run)

确保相机已连接且 IP 可达，RTSP 服务器已启动（如 ZLMediaKit），然后运行脚本：

```bash
cd 海康工业相机推流
python 海康.py
```

### 5. 常见问题 (FAQ)

- **报错 `ImportError: No module named 'MvImport'`**: 
  - 请确保你在 `海康.py` 所在的目录下运行脚本，或者将 `MvImport` 所在的路径添加到 `PYTHONPATH`。
- **FFmpeg 报错**: 
  - 检查 RTSP 服务器是否开启。
  - 检查 `RTSP_URL` 是否正确。
  - 如果没有 NVIDIA 显卡，请将 `USE_NVIDIA_GPU` 设置为 `False`。



## 推流命令

本地视频流推到流媒体服务器：

**注："stream1.mp4" 改成本地视频，"rtsp://127.0.0.1:9994/live/stream1" 改成自定义的流媒体地址（别人可通过此地址访问该视频流）/live/stream1 两层地址是必要的**

```bash
ffmpeg -stream_loop -1 -re -i "stream1.mp4" -an -c:v copy -f rtsp "rtsp://127.0.0.1:9994/live/stream1"
```

## 查看本地硬件的视频流

显示能用的 摄像头

```bash
ffmpeg -list_devices true -f dshow -i dummy
```

本地浏览，通过设置 video=“”，此值是根据上面那条显示可用摄像头命令查询出来的

```bash
ffplay -f dshow -vcodec mjpeg -video_size 1920x1080 -i video="@device_pnp_\\?\usb#vid_2ce3&pid_3894&mi_00#7&18326a43&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
```
