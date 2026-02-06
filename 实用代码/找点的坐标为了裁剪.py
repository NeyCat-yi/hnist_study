import cv2
import numpy as np

# 读取视频
cap = cv2.VideoCapture("rtsp://100.117.27.14:9994/live/stream1")
assert cap.isOpened(), "Error reading video file"

# 读取第一帧作为标注背景
ret, frame = cap.read()
if not ret:
    raise RuntimeError("无法读取视频帧")

points = []  # 存放点击的点坐标

# 鼠标点击事件
def mouse_callback(event, x, y, flags, param):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        points.append((x, y))
        print(f"点 {len(points)}: {(x, y)}")

        # 在图上画出点
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
        cv2.imshow("Select Region", frame)

# 创建窗口并绑定回调
cv2.namedWindow("Select Region")
cv2.setMouseCallback("Select Region", mouse_callback)

print("请用鼠标点击视频中的区域顶点（顺时针或逆时针点四个点）。")
print("按 ESC 退出，或点击完成后关闭窗口。")

while True:
    cv2.imshow("Select Region", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 退出
        break

cv2.destroyAllWindows()
cap.release()

# 打印结果
if points:
    print("\n复制到代码里的 region_points 格式：")
    print({
        "region-01": points
    })
else:
    print("未选择任何点")
