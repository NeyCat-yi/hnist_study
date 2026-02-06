import cv2
from ultralytics import YOLO, solutions
import ultralytics
print(f"Ultralytics Version: {ultralytics.__version__}")
# 查看 ObjectCounter 到底有哪些方法可用
from ultralytics.solutions import ObjectCounter
print(dir(ObjectCounter))
# 1. 加载模型（只加载一次）
model = YOLO(r"D:\code\QT_Fish\res\best_lxf_11_27.pt")

cap = cv2.VideoCapture(r"F:\1月10日 (1)-1.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_multi_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# 定义三个区域
regions_list = [
    [(1281, 347), (1283, 382)],  # Line 1
    [(1452, 350), (1454, 376)],  # Line 2
]

# 2. 初始化计数器（注意：这里不再传入模型路径，而是初始化计数逻辑）
counters = []
for region in regions_list:
    ct = solutions.ObjectCounter(
        show=False,
        region=region,
        # model 参数在某些版本可以不传，或者传入共用的 model
        model=r"D:\code\QT_Fish\res\best_lxf_11_27.pt", 
        conf=0.5,
        iou=0.5,
        show_conf=False,
        show_labels=False
    )
    counters.append(ct)



while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    # 1. 统一进行一次追踪推理（这一步最耗时，只做一次）
    # persist=True 必须加上，否则 ID 会每一帧都变
    tracks = model.track(im0, persist=True, show=False, conf=0.5, iou=0.5)

    # 2. 遍历每个计数器进行处理
    for counter in counters:
        # 关键点：将单次推理的结果 tracks 传入每一个 counter
        # 在最新 API 中，使用 count_objects 处理并返回绘制好的图像
        im0 = counter.count_objects(im0, tracks)

    # --- 以下是 UI 绘制逻辑 ---
    # 统计数据（注意：最新版属性名为 in_count 和 out_count）
    line1_in = counters[0].in_count
    line2_in = counters[1].in_count
    
    passage_rate = (line2_in / line1_in * 100) if line1_in > 0 else 0
    
    # 绘制半透明背景装饰
    cv2.rectangle(im0, (20, 20), (500, 200), (0, 0, 0), -1)
    cv2.putText(im0, f"Passage Rate: {passage_rate:.1f}%", (40, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    for i, ct in enumerate(counters):
        cv2.putText(im0, f"L{i+1} In:{ct.in_count} Out:{ct.out_count}", (40, 110 + i*40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Fish Counting System", im0)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()