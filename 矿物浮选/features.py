import cv2
import numpy as np
from .segmentation import FrothSegmentation
from .utils import calc_stat, Stat
from collections import defaultdict


# 特征结构体
class ColorFeature:
    """颜色特征结构体"""

    def __init__(self, r_mean, g_mean, b_mean, loadrate):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean
        self.loadrate = loadrate


class MorphologicalFeature:
    """形态学特征结构体"""

    def __init__(self, area_mean, area_std, long_diameter_mean, eccentricity_mean):
        self.area_mean = area_mean
        self.area_std = area_std
        self.long_diameter_mean = long_diameter_mean
        self.eccentricity_mean = eccentricity_mean


class TextureFeature:
    """纹理特征结构体"""

    def __init__(self, contrast, energy, entropy, homogenity):
        self.contrast = contrast
        self.energy = energy
        self.entropy = entropy
        self.homogenity = homogenity


class DynamicFeature:
    """动态特征结构体"""

    def __init__(self, velocity_mean, velocity_std, angle_mean, stability):
        self.velocity_mean = velocity_mean
        self.velocity_std = velocity_std
        self.angle_mean = angle_mean
        self.stability = stability


class AllFeature:
    """所有特征结构体"""

    def __init__(
        self,
        r_mean,
        g_mean,
        b_mean,
        loadrate,
        area_mean,
        area_std,
        long_diameter_mean,
        eccentricity_mean,
        velocity_mean,
        velocity_std,
        angle_mean,
        stability,
    ):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean
        self.loadrate = loadrate
        self.area_mean = area_mean
        self.area_std = area_std
        self.long_diameter_mean = long_diameter_mean
        self.eccentricity_mean = eccentricity_mean
        self.velocity_mean = velocity_mean
        self.velocity_std = velocity_std
        self.angle_mean = angle_mean
        self.stability = stability


class IMGStaticFeature:
    """静态图像特征提取类"""

    def __init__(self, img):
        """初始化静态特征提取类

        Args:
            img: 输入的BGR格式图像
        """
        self.img = img
        self.fs = FrothSegmentation(img)
        self.cf = None
        self.mf = None
        self.tf = None

    def get_color_feature(self):
        """获取颜色特征

        Returns:
            ColorFeature: 颜色特征对象
        """
        # 分割RGB通道
        b_channel, g_channel, r_channel = cv2.split(self.img)

        # 计算RGB均值
        r_mean = float(np.mean(np.array(r_channel, dtype=np.float64)))
        g_mean = float(np.mean(np.array(g_channel, dtype=np.float64)))
        b_mean = float(np.mean(np.array(b_channel, dtype=np.float64)))

        # 承载率（1-高亮区域占比）
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, th_high_light = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        highlight_area = cv2.countNonZero(th_high_light)
        gray_area = gray_img.size
        loadrate = 1 - highlight_area / gray_area if gray_area > 0 else 0

        self.cf = ColorFeature(r_mean, g_mean, b_mean, loadrate)
        return self.cf

    def get_morphological_feature(self):
        """获取形态学特征

        Returns:
            MorphologicalFeature: 形态学特征对象
        """
        rcn = self.fs.get_result_coord_num()
        rs = self.fs.get_result_set()
        rm = self.fs.get_result_map()

        # 泡沫面积集合
        areas = []
        # 泡沫尺寸（长径）集合
        long_diameters = []
        short_diameters = []
        # 离心率集合
        eccentricities = []

        for item in rs:
            # 面积
            area = rcn[item]
            areas.append(area)

            # 尺寸
            mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)
            for coord in rm[item]:
                mask[coord[0], coord[1]] = 255

            # 通过椭圆拟合得长短径
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if len(contour) >= 5:  # fitEllipse requires at least 5 points
                    fitted_ellipse = cv2.fitEllipse(contour)
                    short_diameter = min(fitted_ellipse[1][0], fitted_ellipse[1][1])
                    long_diameter = max(fitted_ellipse[1][0], fitted_ellipse[1][1])
                    short_diameters.append(short_diameter)
                    long_diameters.append(long_diameter)

                    # 形状（离心率）
                    eccentricity = (
                        np.sqrt(1 - (short_diameter / long_diameter) ** 2)
                        if long_diameter > 0
                        else 0
                    )
                    eccentricities.append(eccentricity)

        # 由集合得到统计量
        areas_stat = calc_stat(areas)
        long_diameter_stat = (
            calc_stat(long_diameters) if long_diameters else Stat(0, 0, 0, 0)
        )
        eccentricity_stat = (
            calc_stat(eccentricities) if eccentricities else Stat(0, 0, 0, 0)
        )

        self.mf = MorphologicalFeature(
            areas_stat.mean,
            areas_stat.std_dev,
            long_diameter_stat.mean,
            eccentricity_stat.mean,
        )
        return self.mf

    def get_texture_feature(self):
        """获取纹理特征（当前未实现）

        Returns:
            TextureFeature: 纹理特征对象
        """
        # 暂时不可用，返回默认值
        self.tf = TextureFeature(0, 0, 0, 0)
        return self.tf


class VideoFeature:
    """视频特征提取类"""

    def __init__(self, video_path):
        """初始化视频特征提取类

        Args:
            video_path: 视频文件路径
        """
        self.cap = cv2.VideoCapture(video_path)
        self.final_frame_isf = None

        # 获取总帧数
        frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 读取最后一帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frames_count - 1)
        ret, final_frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read the last frame from video")

        self.final_frame_isf = IMGStaticFeature(final_frame)
        self.cap.release()

    def get_img_static_feature(self):
        """获取最后一帧的静态特征

        Returns:
            IMGStaticFeature: 静态特征对象
        """
        return self.final_frame_isf


class MultiImgFeature:
    """多图像特征提取类"""

    def __init__(self, imgs):
        """初始化多图像特征提取类

        Args:
            imgs: 图像列表，按时间顺序排列
        """
        self.multi_img = imgs
        self.img_count = len(imgs)
        self.final_frame_isf = None
        self.final_frame_df = None

    def get_dynamic_feature(self):
        """获取动态特征

        Returns:
            DynamicFeature: 动态特征对象
        """
        if len(self.multi_img) < 2:
            raise ValueError(
                "At least two images are required for dynamic feature extraction"
            )

        final_frame = self.multi_img[-1]  # 目标图
        previous_frame = self.multi_img[-2]  # 参考图

        # 子块配对结果变量
        pair_markers = {}
        pair_similarities = {}

        final_frame_fs = FrothSegmentation(final_frame)
        previous_frame_fs = FrothSegmentation(previous_frame)

        final_frame_rs = final_frame_fs.get_result_set()
        final_frame_rcn = final_frame_fs.get_result_coord_num()
        final_frame_rm = final_frame_fs.get_result_map()
        previous_frame_markers = previous_frame_fs.get_result()
        previous_frame_rcn = previous_frame_fs.get_result_coord_num()
        previous_frame_rm = previous_frame_fs.get_result_map()

        # 遍历有效marker
        for item in final_frame_rs:
            # 使用 defaultdict 统计marker的重复个数
            marker_counts = defaultdict(int)
            for coord in final_frame_rm[item]:
                if previous_frame_markers is not None:
                    marker = previous_frame_markers[coord[0], coord[1]]
                else:
                    marker = 0
                marker_counts[str(marker)] += 1

            # 寻找具有最大重复次数的元素和对应的次数
            max_marker = ""
            max_count = 0
            for marker, count in marker_counts.items():
                if count > max_count:
                    max_marker = marker
                    max_count = count

            # 填写结果
            pair_markers[item] = max_marker
            pair_similarities[item] = (
                max_count / final_frame_rcn[item] if final_frame_rcn[item] > 0 else 0
            )

        # 自适应不稳定阈值
        similarities = list(pair_similarities.values())
        simi_stat = calc_stat(similarities)
        unstable_lb = simi_stat.mean - simi_stat.std_dev

        unstable_count = 0  # 不稳定子块数
        block_velo = {}  # <子块索引，速度大小，方向>
        for item, max_marker in pair_markers.items():
            if pair_similarities[item] < unstable_lb:
                unstable_count += 1
            else:
                if (
                    max_marker in previous_frame_rm
                    and previous_frame_rcn[max_marker] > 0
                ):
                    # 计算目标块坐标
                    coord1 = (
                        sum([coord[0] for coord in final_frame_rm[item]])
                        / final_frame_rcn[item]
                    )
                    coord2 = (
                        sum([coord[1] for coord in final_frame_rm[item]])
                        / final_frame_rcn[item]
                    )
                    coord_tar_block = (coord1, coord2)

                    # 计算参考块坐标
                    coord1 = (
                        sum([coord[0] for coord in previous_frame_rm[max_marker]])
                        / previous_frame_rcn[max_marker]
                    )
                    coord2 = (
                        sum([coord[1] for coord in previous_frame_rm[max_marker]])
                        / previous_frame_rcn[max_marker]
                    )
                    coord_ref_block = (coord1, coord2)

                    # 根据坐标算相对位置（速度）
                    velocity = np.sqrt(
                        (coord_tar_block[0] - coord_ref_block[0]) ** 2
                        + (coord_tar_block[1] - coord_ref_block[1]) ** 2
                    )

                    # 计算角度
                    dx = coord_tar_block[0] - coord_ref_block[0]
                    dy = coord_tar_block[1] - coord_ref_block[1]
                    if dx == 0:
                        angle = np.pi / 2 if dy >= 0 else -np.pi / 2
                    else:
                        angle = np.arctan2(dy, dx)

                    # 记录结果
                    block_velo[item] = (velocity, angle)

        velocities = [velo[0] for velo in block_velo.values()]
        angles = [velo[1] for velo in block_velo.values()]

        velocities_stat = calc_stat(velocities) if velocities else Stat(0, 0, 0, 0)
        angles_stat = calc_stat(angles) if angles else Stat(0, 0, 0, 0)

        self.final_frame_df = DynamicFeature(
            velocities_stat.mean if not np.isnan(velocities_stat.mean) else -1,
            velocities_stat.std_dev if not np.isnan(velocities_stat.std_dev) else -1,
            angles_stat.mean if not np.isnan(angles_stat.mean) else -1,
            1 - unstable_count / len(pair_markers) if pair_markers else 0,
        )
        return self.final_frame_df

    def get_all_feature(self):
        """获取所有特征

        Returns:
            AllFeature: 所有特征对象
        """
        final_frame = self.multi_img[-1]
        self.final_frame_isf = IMGStaticFeature(final_frame)
        cf = self.final_frame_isf.get_color_feature()
        mf = self.final_frame_isf.get_morphological_feature()
        tf = self.final_frame_isf.get_texture_feature()
        df = self.get_dynamic_feature()

        return AllFeature(
            cf.r_mean,
            cf.g_mean,
            cf.b_mean,
            cf.loadrate,
            mf.area_mean,
            mf.area_std,
            mf.long_diameter_mean,
            mf.eccentricity_mean,
            df.velocity_mean,
            df.velocity_std,
            df.angle_mean,
            df.stability,
        )
