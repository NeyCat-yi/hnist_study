import cv2
import numpy as np

def concat_three_videos(video1_path, video2_path, video3_path, output_path, mode="horizontal"):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    cap3 = cv2.VideoCapture(video3_path)

    if not cap1.isOpened() or not cap2.isOpened() or not cap3.isOpened():
        print("无法打开其中一个视频文件")
        return

    # 获取 fps（取最小值以保证同步）
    fps = min(
        cap1.get(cv2.CAP_PROP_FPS),
        cap2.get(cv2.CAP_PROP_FPS),
        cap3.get(cv2.CAP_PROP_FPS)
    )

    # 获取分辨率
    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w3, h3 = int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if mode == "horizontal":
        # 统一高度
        new_h = min(h1, h2, h3)
        s1, s2, s3 = new_h / h1, new_h / h2, new_h / h3
        new_w1, new_w2, new_w3 = int(w1 * s1), int(w2 * s2), int(w3 * s3)
        out_size = (new_w1 + new_w2 + new_w3, new_h)
    else:  # vertical
        # 统一宽度
        new_w = min(w1, w2, w3)
        s1, s2, s3 = new_w / w1, new_w / w2, new_w / w3
        new_h1, new_h2, new_h3 = int(h1 * s1), int(h2 * s2), int(h3 * s3)
        out_size = (new_w, new_h1 + new_h2 + new_h3)

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    while True:
        r1, f1 = cap1.read()
        r2, f2 = cap2.read()
        r3, f3 = cap3.read()
        if not r1 or not r2 or not r3:
            break

        if mode == "horizontal":
            f1 = cv2.resize(f1, (new_w1, new_h))
            f2 = cv2.resize(f2, (new_w2, new_h))
            f3 = cv2.resize(f3, (new_w3, new_h))
            frame = np.hstack((f1, f2, f3))
        else:
            f1 = cv2.resize(f1, (new_w, new_h1))
            f2 = cv2.resize(f2, (new_w, new_h2))
            f3 = cv2.resize(f3, (new_w, new_h3))
            frame = np.vstack((f1, f2, f3))

        out.write(frame)
        cv2.imshow("拼接结果", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap1.release()
    cap2.release()
    cap3.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 拼接完成，输出文件: {output_path}")


if __name__ == "__main__":
    v1 = r"masked_output1.mp4"
    v2 = r"G:\hl\huanglei\fishroad_data\9.2videos\cropped.mp4"
    v3 = r"masked_output3.mp4"
    out = "concat_3videos.mp4"

    # mode 可选 "horizontal" 或 "vertical"
    concat_three_videos(v1, v2, v3, out, mode="horizontal")
