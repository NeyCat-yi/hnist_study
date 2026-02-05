# -*- coding: utf-8 -*-
import sys
import threading
import msvcrt
import numpy as np
import cv2
import subprocess

# 引入海康SDK库 (确保 MvImport 文件夹在当前目录)
sys.path.append("./MvImport")
from MvImport.MvCameraControl_class import *

# ================= 配置区域 =================
TARGET_IP = "169.254.211.253"  # 你的相机IP
RTSP_URL = "rtsp://127.0.0.1:9994/live/stream"  # 推流地址 (请修改为你的服务器地址)
FPS = 25  # 推流帧率 (建议与相机采集帧率一致)
USE_NVIDIA_GPU = True  # 是否使用 NVIDIA 显卡硬件加速 (True: 开启, False: 使用 CPU)


# ===========================================

def find_device_by_ip(ip):
    """ 根据IP查找设备信息 (修正版) """
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # 枚举设备
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print(f"枚举设备失败! ret=0x{ret:x}")
        return None

    if deviceList.nDeviceNum == 0:
        print("未发现任何设备")
        return None

    print(f"发现 {deviceList.nDeviceNum} 个设备，正在匹配 IP: {ip} ...")

    for i in range(0, deviceList.nDeviceNum):
        # 将指针转换为 Python 结构体对象
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents

        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            # 获取GigE设备信息
            strIp = ""
            # 解析 IP 地址
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            strIp = f"{nip1}.{nip2}.{nip3}.{nip4}"

            if strIp == ip:
                print(f"找到目标设备: {strIp}")
                # [关键修改] 这里必须返回 .contents (结构体本身)，不能返回 .pDeviceInfo[i] (指针)
                return mvcc_dev_info

    print(f"未找到IP为 {ip} 的设备")
    return None


def main():
    # 1. 查找设备
    stDeviceList = find_device_by_ip(TARGET_IP)
    if stDeviceList is None:
        return

    # 2. 创建相机实例
    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print(f"创建句柄失败 ret=0x{ret:x}")
        return

    # 3. 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"打开设备失败 ret=0x{ret:x}")
        return

    # 4. 配置参数 (可选：设置触发模式为Off，即连续采集)
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    # 获取图像宽高 (用于配置FFmpeg)
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    ret = cam.MV_CC_GetIntValue("Width", stParam)
    nWidth = stParam.nCurValue
    ret = cam.MV_CC_GetIntValue("Height", stParam)
    nHeight = stParam.nCurValue

    print(f"相机分辨率: {nWidth}x{nHeight}")

    # 需要分配足够的内存用于像素格式转换 (转为BGR24，每像素3字节)
    nDataSize = nWidth * nHeight * 3
    pDataForRGB = (c_ubyte * nDataSize)()
    stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()  # 转换参数结构体

    # 5. 启动 FFmpeg 管道
    
    # 动态配置编码器选项
    encoder_options = []
    if USE_NVIDIA_GPU:
        print(">> 使用 NVIDIA GPU 硬件加速编码")
        encoder_options = [
            '-c:v', 'h264_nvenc',      # 使用 NVIDIA 显卡编码
            '-pix_fmt', 'yuv420p',
            '-preset', 'llhp',         # 低延迟高性能预设
            '-b:v', '8000k',           # 码率 (根据分辨率调整)
            '-maxrate', '10000k',
            '-bufsize', '20000k'
        ]
    else:
        print(">> 使用 CPU 软件编码 (libx264)")
        encoder_options = [
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',    # 极速模式，降低延迟
            '-tune', 'zerolatency',    # 零延迟调优
            '-b:v', '4000k'            # 码率 (CPU编码通常需要更低码率以减轻负载)
        ]

    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{nWidth}x{nHeight}',
        '-r', str(FPS),
        '-i', '-',
        
        # 插入编码选项
        *encoder_options,
        
        '-vf', 'scale=1920:-2',  # 缩放到 1920 宽 (高度自适应) 以保证流畅性
        '-f', 'rtsp',
        RTSP_URL
    ]

    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

    # --- 新增：自动调整网络包大小 (解决 GigE 相机丢包导致取不到图的核心问题) ---
    nPacketSize = cam.MV_CC_GetOptimalPacketSize()
    if int(nPacketSize) > 0:
        ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
        if ret != 0:
            print(f"警告: 设置数据包大小失败 ret[0x{ret:x}]")
    else:
        print("警告: 获取最佳包大小失败")

    # 6. 开始采集
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"开始采集失败 ret=0x{ret:x}")
        return

    data_buf = (c_ubyte * nDataSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    print("\n=== 开始进入采集循环 (Debug模式) ===")
    frame_count = 0

    try:
        while True:
            # 获取一帧原始数据
            pData = byref(data_buf)
            # 这里的 1000 是超时时间 (ms)
            ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)

            if ret == 0:
                # 成功取到一帧
                frame_count += 1
                if frame_count % 25 == 0:  # 每25帧打印一次，避免刷屏
                    print(
                        f"成功获取第 {frame_count} 帧 | 分辨率: {stFrameInfo.nWidth}x{stFrameInfo.nHeight} | 格式: 0x{stFrameInfo.enPixelType:x}")

                # --- 像素格式转换 ---
                memset(byref(stConvertParam), 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM))
                stConvertParam.nWidth = stFrameInfo.nWidth
                stConvertParam.nHeight = stFrameInfo.nHeight
                stConvertParam.pSrcData = data_buf
                stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
                stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
                stConvertParam.pDstBuffer = pDataForRGB
                stConvertParam.nDstBufferSize = nDataSize

                ret_conv = cam.MV_CC_ConvertPixelType(stConvertParam)

                if ret_conv == 0:
                    try:
                        # 写入 FFmpeg 管道
                        pipe.stdin.write(string_at(pDataForRGB, stConvertParam.nDstLen))
                    except Exception as e:
                        print(f"管道写入错误: {e}")
                        break
                else:
                    print(f"Error: 像素转换失败 ret=0x{ret_conv:x}")
            else:
                # 打印具体的错误代码
                print(f"Error: 获取图像超时/失败 ret=0x{ret:x}")

    except KeyboardInterrupt:
        print("停止推流...")
    finally:
        # 清理资源
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        pipe.terminate()


if __name__ == "__main__":
    main()