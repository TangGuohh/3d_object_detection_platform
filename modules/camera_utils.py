import json
import math
import numpy as np
from PIL import Image
import os

class CameraUtils:
    @staticmethod
    def load_camera_params(image_name, json_path='./spatial_understanding/cam_infos.json'):
        """
        从JSON文件中加载指定图像的相机参数

        参数:
            image_name (str): 图像文件名，用作查找相机参数的键
            json_path (str): 包含相机参数的JSON文件路径，默认为'./spatial_understanding/cam_infos.json'

        返回:
            dict or None: 成功时返回对应图像的相机参数字典，失败或未找到时返回None
        """
        try:
            # 检查JSON文件是否存在
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    cam_infos = json.load(f)
                # 返回指定图像的相机参数，如果不存在则返回None
                return cam_infos.get(image_name, None)
            return None
        except Exception as e:
            print(f"加载相机参数失败: {e}")
            return None

    @staticmethod
    def generate_camera_params(image_path, fx=None, fy=None, cx=None, cy=None, fov=60):
        """
        根据图像尺寸和可选的相机内参生成相机参数

        参数:
            image_path (str): 图像文件路径，用于获取图像尺寸
            fx (float): 相机x轴焦距，如果为None则根据视场角计算
            fy (float): 相机y轴焦距，如果为None则根据视场角计算
            cx (float): 相机光心x坐标，如果为None则设为图像宽度的一半
            cy (float): 相机光心y坐标，如果为None则设为图像高度的一半
            fov (int): 相机视场角（度），默认为60度，用于计算焦距

        返回:
            dict: 包含相机内参的字典，包括fx, fy, cx, cy四个参数
        """
        try:
            # 打开图像并获取尺寸
            image = Image.open(image_path)
            w, h = image.size

            # 如果未提供焦距参数，则根据视场角和图像尺寸计算
            if fx is None or fy is None:
                # 使用宽高的平均值来计算焦距，让fx和fy相同
                avg_dimension = (w + h) / 2
                focal_length = round(avg_dimension / (2 * np.tan(np.deg2rad(fov) / 2)), 2)
                fx = focal_length
                fy = focal_length

            # 如果未提供光心坐标，则设置为图像中心
            if cx is None or cy is None:
                cx = round(w / 2, 2)
                cy = round(h / 2, 2)

            return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        except Exception as e:
            print(f"生成相机参数失败: {e}")
            # 返回默认参数
            return {'fx': 1000, 'fy': 1000, 'cx': 640, 'cy': 360}