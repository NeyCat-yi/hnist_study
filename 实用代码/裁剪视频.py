import cv2

def crop_video(input_path, output_path, x, y, w, h):
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频")
        return

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"输入视频: {width}x{height}, {fps:.2f}fps, {total_frames}帧")

    # 定义输出视频编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 也可以用 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
  
    # 逐帧读取并裁剪
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 裁剪区域
        cropped = frame[y:y+h, x:x+w]

        # 写入输出
        out.write(cropped)

        # 显示裁剪结果（按 q 退出预览）
        cv2.imshow("裁剪结果", cropped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"裁剪完成，输出文件: {output_path}")


if __name__ == "__main__":
    # 输入视频路径
    input_video = r"G:\hl\huanglei\fishroad_data\9.2videos\jy_01video_undistort1.mp4"
    output_video = "cropped1.mp4"

    # 手动指定裁剪区域 (x, y, w, h)
    # 例如：从 (100, 50) 开始，宽 400，高 300
    crop_x, crop_y, crop_w, crop_h = 10, 10, 1910, 700

    crop_video(input_video, output_video, crop_x, crop_y, crop_w, crop_h)
