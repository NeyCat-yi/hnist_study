import cv2
import numpy as np
from collections import defaultdict


class FrothSegmentation:
    def __init__(self, img):
        """初始化泡沫分割类，执行泡沫图像分割

        Args:
            img: 输入的BGR格式图像
        """
        self.img = img
        self.markers = None
        self.marker_map = defaultdict(list)  # 由marker映射到所有对应像素坐标的结构
        self.marker_set = []  # 记录所有marker种类
        self.marker_coord_num = {}  # 由marker映射到所有对应像素个数的结构

        # 灰度化
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 中值滤波
        blur_img = cv2.medianBlur(gray_img, 7)
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_img = clahe.apply(blur_img)
        # 阈值分割，得到泡沫高亮标记
        _, th_high_light = cv2.threshold(
            cl_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        # 得到边缘
        _, th_black_line = cv2.threshold(cl_img, 30, 255, cv2.THRESH_BINARY)
        th_white_line = cv2.bitwise_not(th_black_line)

        # 高亮标记滤波
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th_high_light_open = cv2.morphologyEx(
            th_high_light, cv2.MORPH_OPEN, kernel, iterations=2
        )
        th_high_light_filter = cv2.morphologyEx(
            th_high_light_open, cv2.MORPH_DILATE, kernel, iterations=2
        )

        # 求标记的连通域
        _, markers = cv2.connectedComponents(th_high_light_filter)
        # 转换为有符号整数，因为watershed需要
        markers = markers.astype(np.int32)

        # 为watershed准备标记
        markers = markers + 1
        # 将未知区域标记为0
        unknown = cv2.subtract(th_black_line, th_high_light_filter)
        markers[unknown == 255] = 0

        # 使用分水岭算法
        self.markers = cv2.watershed(img, markers)

        # 遍历结果生成marker的各种形式
        for i in range(self.markers.shape[0]):
            for j in range(self.markers.shape[1]):
                marker = self.markers[i, j]
                marker_str = str(marker)
                # 添加坐标
                self.marker_map[marker_str].append((i, j))
                # 记录新种类
                if marker_str not in self.marker_set:
                    self.marker_set.append(marker_str)
                    self.marker_coord_num[marker_str] = 1
                else:
                    self.marker_coord_num[marker_str] += 1

        # 对marker_set处理：去掉-1和过小的marker
        area_limit = 100
        self.marker_set = [
            item
            for item in self.marker_set
            if item != "-1" and self.marker_coord_num[item] >= area_limit
        ]

    def get_result(self):
        """获取分割结果

        Returns:
            cv2.Mat: 分割标记图像
        """
        return self.markers

    def get_result_coord_num(self):
        """获取各marker的像素数量

        Returns:
            dict: marker字符串到像素数量的映射
        """
        return self.marker_coord_num

    def get_result_set(self):
        """获取有效的marker集合

        Returns:
            list: 有效的marker字符串列表
        """
        return self.marker_set

    def get_result_map(self):
        """获取各marker对应的像素坐标

        Returns:
            dict: marker字符串到坐标列表的映射
        """
        return self.marker_map
