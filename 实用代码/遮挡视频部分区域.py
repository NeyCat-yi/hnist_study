import cv2
import numpy as np

# 全局变量
drawing = False
ix, iy = -1, -1
rectangles = []  # 存储多个矩形区域
current_rect = None  # 当前正在绘制的矩形

def draw_rectangles(event, x, y, flags, param):
    global ix, iy, drawing, rectangles, current_rect, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_rect = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 复制原始帧，确保每次绘制都是在干净的基础上
            frame_copy = frame.copy()
            # 绘制已保存的所有矩形
            for rect in rectangles:
                x1, y1, w, h = rect
                cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            # 绘制当前正在拖拽的矩形
            current_rect = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
            cv2.rectangle(frame_copy, (current_rect[0], current_rect[1]),
                          (current_rect[0] + current_rect[2], current_rect[1] + current_rect[3]), 
                          (0, 0, 255), 2)  # 红色表示当前正在绘制的矩形

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if current_rect is not None and current_rect[2] > 0 and current_rect[3] > 0:
            rectangles.append(current_rect)
        current_rect = None
        # 重新绘制所有矩形
        frame_copy = frame.copy()
        for rect in rectangles:
            x1, y1, w, h = rect
            cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

def mask_video_regions(video_path, output_path, color=(0, 0, 0)):
    global frame, frame_copy, rectangles
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取第一帧用于选择区域
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return

    frame_copy = frame.copy()
    cv2.namedWindow("选择遮挡区域 (拖动鼠标选择，可多选，完成后按s键)")
    cv2.setMouseCallback("选择遮挡区域 (拖动鼠标选择，可多选，完成后按s键)", draw_rectangles)

    while True:
        cv2.imshow("选择遮挡区域 (拖动鼠标选择，可多选，完成后按s键)", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # 按 q 退出
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord("s") and len(rectangles) > 0:  # 按 s 保存并开始处理，确保至少选择了一个区域
            break
        elif key == ord("d"):  # 按 d 删除最后一个选择的区域
            if rectangles:
                rectangles.pop()
                # 重新绘制所有矩形
                frame_copy = frame.copy()
                for rect in rectangles:
                    x1, y1, w, h = rect
                    cv2.rectangle(frame_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

    # 重置视频到开头
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 显示选择的区域信息
    print(f"共选择了 {len(rectangles)} 个遮挡区域:")
    for i, rect in enumerate(rectangles, 1):
        x, y, w, h = rect
        print(f"区域 {i}: x={x}, y={y}, w={w}, h={h}")

    # 处理视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 对每个选择的区域进行遮挡
        for rect in rectangles:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=-1)

        out.write(frame)
        cv2.imshow("处理后视频", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 清理资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成，输出文件: {output_path}")


if __name__ == "__main__":
    video = "cropped1.mp4"  # 输入视频路径
    output = "masked_output1.mp4"  # 输出视频路径
    mask_video_regions(video, output, color=(255, 255, 255))  # 白色遮挡，可以改为其他颜色如(0,0,0)黑色
