import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json

class Visualization2D:
    """
    用于在图像上绘制二维边界框（bounding boxes）和对应标签的可视化工具类
    """

    def __init__(self):
        """
        初始化函数，设置中文字体支持以确保中文显示正常
        """
        self._setup_chinese_font()

    def _setup_chinese_font(self):
        """
        配置matplotlib使用支持中文的字体，并关闭unicode负号替换功能
        """
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def draw_bboxes_2d(self, image_path, bbox_data, output_path=None):
        """
        在指定图像上绘制2D边界框及标签信息，并可选择性地将结果保存为文件

        参数:
            image_path (str): 输入图像的路径
            bbox_data (list or str): 边界框数据，可以是字典列表、坐标列表或JSON字符串形式
            output_path (str, optional): 输出图像的保存路径，默认为None表示不保存

        返回:
            matplotlib.figure.Figure: 包含绘图结果的Figure对象，可用于进一步操作或展示
        """
        try:
            # 读取输入图像
            image = Image.open(image_path)
            raw_width, raw_height = image.size

            # 创建绘图画布
            fig, ax = plt.subplots(1, figsize=(12, 8))

            # 将图像显示在画布上
            ax.imshow(image)

            # 定义一组颜色用于区分不同边界框
            colors = [
                '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
                '#FFA500', '#800080', '#008000', '#FFC0CB', '#FFD700', '#4B0082',
                '#00FF7F', '#DC143C', '#8A2BE2', '#7CFC00', '#FF4500', '#DA70D6',
                '#20B2AA', '#FF69B4', '#32CD32', '#BA55D3', '#9370DB', '#3CB371'
            ]

            # 存储处理后的边界框信息
            processed_bboxes = []

            # 如果输入的是字符串，则尝试将其解析为JSON格式的数据
            if isinstance(bbox_data, str):
                try:
                    bbox_data = json.loads(bbox_data)
                except json.JSONDecodeError:
                    print("无法解析JSON字符串")
                    return fig

            # 处理列表类型的边界框数据
            if isinstance(bbox_data, list):
                for item in bbox_data:
                    if isinstance(item, dict):
                        # 判断是否存在有效的边界框字段并提取相关信息
                        bbox_key = None
                        if 'bbox_2d' in item:
                            bbox_key = 'bbox_2d'
                        elif 'bbox' in item:
                            bbox_key = 'bbox'
                        elif 'bounding_box' in item:
                            bbox_key = 'bounding_box'

                        if bbox_key and item.get(bbox_key):
                            processed_bboxes.append({
                                "bbox": item[bbox_key],
                                "label": item.get("label", "unknown"),
                                "score": item.get("score", 1.0)
                            })
                    elif isinstance(item, list) and len(item) >= 4:
                        # 若为直接坐标列表，则默认标签为"object"
                        processed_bboxes.append({
                            "bbox": item,
                            "label": "object",
                            "score": 1.0
                        })

            print(f"处理后的2D边界框数据: {processed_bboxes}")

            # 遍历所有处理好的边界框进行绘制
            for i, item in enumerate(processed_bboxes):
                bbox = item["bbox"]
                label = item["label"]
                score = item["score"]
                color = colors[i % len(colors)]

                # 根据边界框坐标的数值范围判断其是否需要转换成像素单位
                if len(bbox) == 4:
                    if bbox[2] > 1 and bbox[3] > 1:  # 假设是绝对坐标
                        x1, y1, x2, y2 = bbox
                    else:  # 可能是归一化坐标
                        img_w, img_h = image.size
                        x1, y1, x2, y2 = bbox[0]*img_w, bbox[1]*img_h, bbox[2]*img_w, bbox[3]*img_h
                else:
                    print(f"无效的边界框数据: {bbox}")
                    continue

                abs_y1 = int(item["bbox"][1] / 1000 * raw_height)
                abs_x1 = int(item["bbox"][0] / 1000 * raw_width)
                abs_y2 = int(item["bbox"][3] / 1000 * raw_height)
                abs_x2 = int(item["bbox"][2] / 1000 * raw_width)

                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1

                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1

                # 计算边界框的宽度和高度
                width = abs_x2 - abs_x1
                height = abs_y2 - abs_y1

                # 绘制矩形边界框
                rect = patches.Rectangle((abs_x1, abs_y1), width, height,
                                       linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                # 构造要显示的文本内容并在边界框上方添加标签
                label_text = f"{label}: {score:.2f}" if score < 1.0 else label
                ax.text(abs_x1 + 8, abs_y1 + 6, label_text,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                       color='white', fontsize=12, weight='bold')

            # 设置图表标题并隐藏坐标轴刻度
            #ax.set_title('2D目标检测结果', fontsize=16)
            ax.axis('off')

            # 如有指定输出路径则保存图像至该位置
            if output_path:
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"结果已保存到: {output_path}")

            return fig

        except Exception as e:
            print(f"绘制2D边界框时出错: {e}")
            # 出现异常时返回一个包含错误提示的空白图像
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.text(0.5, 0.5, f"可视化错误: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return fig