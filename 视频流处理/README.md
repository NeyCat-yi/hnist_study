# 视频流处理

## 海康视频流处理

1、下载 MVS（海康工业相机自带软件）

2、获得海康工业相机的IP地址（可以通过 MVS 连接海康工业相机获得）

3、用 MVS 下载目录下自带的海康工业相机 SDK 捕获视频帧然后利用 FFmpeg 命令推流到 流媒体服务器上（这一步用 python 代码完成，可以做视频处理例如：缩放、变形等）**代码见“海康工业相机推流”**

4、（可选）若是需要远程连接流媒体服务器以访问远处的海康工业相机画面，可以使用 **Tailscale** 软件进行异地组网或内网穿透

## 一般视频流处理

1、获得视频帧的画面（可用代码来获取以便于做处理或者FFmpeg调用硬件获取）

2、利用 FFmpeg 推流到 流媒体服务器

3、远程访问同海康视频流处理

## 流媒体服务器

MediaServer 文件夹内以下路径是一个流媒体服务器

```bash
MediaServer\bin\bin.x86.windows10\MediaServer.exe
```

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

