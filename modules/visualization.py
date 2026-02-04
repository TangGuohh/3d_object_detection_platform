# 导入所需的库
import random
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Visualization3D:
    """
    3D可视化类，用于在2D图像上绘制3D边界框
    """
    def __init__(self):
        """
        初始化函数
        设置中文字体支持
        """
        self._setup_chinese_font()

    def _setup_chinese_font(self):
        """
        设置matplotlib支持中文字体
        防止中文显示为方块
        """
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def convert_3dbbox(self, point, cam_params):
        """
        将3D边界框转换为2D图像坐标

        参数:
        point: 3D边界框参数 [x, y, z, x_size, y_size, z_size, pitch, yaw, roll]
        cam_params: 相机参数字典，包含 'fx', 'fy', 'cx', 'cy'

        返回:
        img_corners: 投影到图像上的角点坐标列表
        """
        # 解析3D边界框参数
        x, y, z, x_size, y_size, z_size, pitch, yaw, roll = point

        # 计算边界框的半尺寸
        hx, hy, hz = x_size / 2, y_size / 2, z_size / 2

        # 定义局部坐标系下的8个角点
        local_corners = [
            [ hx,  hy,  hz],   # 0
            [ hx,  hy, -hz],   # 1
            [ hx, -hy,  hz],   # 2
            [ hx, -hy, -hz],   # 3
            [-hx,  hy,  hz],   # 4
            [-hx,  hy, -hz],   # 5
            [-hx, -hy,  hz],   # 6
            [-hx, -hy, -hz]    # 7
        ]

        # 预计算三角函数值以提高效率
        cos_roll = math.cos(roll)
        sin_roll = math.sin(roll)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        img_corners = []

        # 对每个角点进行变换和投影
        for corner in local_corners:
            # 获取局部坐标
            x0, y0, z0 = corner

            # 执行欧拉角旋转: Yaw(偏航) -> Pitch(俯仰) -> Roll(翻滚)

            # Yaw (绕Y轴旋转)
            x1 = x0 * cos_yaw + z0 * sin_yaw
            y1 = y0
            z1 = -x0 * sin_yaw + z0 * cos_yaw

            # Pitch (绕X轴旋转)
            x2 = x1
            y2 = y1 * cos_pitch - z1 * sin_pitch
            z2 = y1 * sin_pitch + z1 * cos_pitch

            # Roll (绕Z轴旋转)
            x3 = x2 * cos_roll - y2 * sin_roll
            y3 = x2 * sin_roll + y2 * cos_roll
            z3 = z2

            # 平移到世界坐标系
            X, Y, Z = x3 + x, y3 + y, z3 + z

            # 透视投影到图像平面
            if Z > 0:  # 只处理相机前方的点
                x_2d = cam_params['fx'] * (X / Z) + cam_params['cx']
                y_2d = cam_params['fy'] * (Y / Z) + cam_params['cy']
                img_corners.append([x_2d, y_2d])

        return img_corners

    def draw_3dbboxes(self, image_path, cam_params, bbox_3d_list):
        """
        在图像上绘制3D边界框并返回matplotlib图形

        参数:
        image_path: 图像文件路径
        cam_params: 相机参数字典
        bbox_3d_list: 3D边界框列表，每个元素可以是字典或列表

        返回:
        fig: matplotlib图形对象，如果出错则返回None
        """
        try:
            # 读取图像
            annotated_image = cv2.imread(image_path)
            if annotated_image is None:
                print(f"无法读取图像: {image_path}")
                return None

            # 定义立方体的12条边（连接角点的索引对）
            edges = [
                [0,1], [2,3], [4,5], [6,7],  # 水平边
                [0,2], [1,3], [4,6], [5,7],  # 垂直边
                [0,4], [1,5], [2,6], [3,7]   # 深度边
            ]

            all_label_positions = []
            # 为每个边界框绘制3D框
            for bbox_data in bbox_3d_list:
                # 处理不同格式的数据输入
                if isinstance(bbox_data, dict) and 'bbox_3d' in bbox_data:
                    bbox_3d = bbox_data['bbox_3d']
                    label = bbox_data.get('label', 'object')
                else:
                    bbox_3d = bbox_data
                    label = 'object'

                # 将边界框参数转换为列表
                bbox_3d = list(bbox_3d)

                # 将角度从度转换为弧度
                pitch_rad = np.deg2rad(bbox_3d[6])
                yaw_rad = np.deg2rad(bbox_3d[7])
                roll_rad = np.deg2rad(bbox_3d[8])

                # 更新角度值为弧度
                bbox_3d[6] = pitch_rad
                bbox_3d[7] = yaw_rad
                bbox_3d[8] = roll_rad

                # 将3D边界框投影到2D图像
                bbox_2d = self.convert_3dbbox(bbox_3d, cam_params)

                # 确保有足够的角点进行绘制
                if len(bbox_2d) >= 8:
                    # 为每个框生成随机颜色
                    box_color = [random.randint(0, 255) for _ in range(3)]

                    # 绘制所有边
                    for start, end in edges:
                        try:
                            pt1 = tuple([int(_pt) for _pt in bbox_2d[start]])
                            pt2 = tuple([int(_pt) for _pt in bbox_2d[end]])
                            cv2.line(annotated_image, pt1, pt2, box_color, 2)

                            # 在第一个角点处添加标签
                            if start == 0:
                                all_label_positions.append((bbox_2d[0][0], bbox_2d[0][1]-10, label, box_color))
                        except:
                            continue

            # 将BGR图像转换为RGB以适应matplotlib显示
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # 创建matplotlib图形
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(annotated_image_rgb)
            for x, y, label_text, color in all_label_positions:
                ax.text(x, y, label_text, fontsize=12, color=np.array(color)/255.0, verticalalignment='bottom')
            ax.axis('off')

            return fig

        except Exception as e:
            print(f"绘制3D边界框时出错: {e}")
            return None