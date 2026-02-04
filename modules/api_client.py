import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class APIClient:
    """
    API客户端类，用于与ModelScope和DashScope等平台的视觉语言模型进行交互。

    属性:
        ms_api_key (str): ModelScope平台的API密钥。
        ms_base_url (str): ModelScope平台的基础URL。
        dash_api_key (str): DashScope平台的API密钥。
        dash_base_url (str): DashScope平台的基础URL。
    """

    def __init__(self):
        self.ms_api_key = "ms-c27f5d0e-00dd-464f-8a69-f5f1474dc54d"
        self.ms_base_url = "https://api-inference.modelscope.cn/v1"
        self.dash_api_key = os.getenv("dash_api_key")
        self.dash_base_url = os.getenv("dash_base_url")

    def encode_image(self, image_path):
        """
        将本地图片转换为Base64编码

        参数:
            image_path (str): 图片文件路径

        返回:
            str: Base64编码后的图片数据
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def inference_with_api(self, image_path, prompt, api_type="modelscope", 
                          high_resolution=False, model_name=None):
        """
        使用指定平台的API进行图像理解推理

        参数:
            image_path (str): 待分析图片的路径
            prompt (str): 提供给模型的问题或指令
            api_type (str): API类型，可选"modelscope"或"dashscope"
            high_resolution (bool): 是否启用高分辨率图像处理模式
            model_name (str): 指定使用的具体模型名称

        返回:
            str: 模型返回的文本结果

        异常:
            FileNotFoundError: 当指定的图片文件不存在时抛出
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        # 根据API类型设置对应的认证信息和默认模型
        if api_type == "modelscope":
            api_key = self.ms_api_key
            base_url = self.ms_base_url
            if model_name is None:
                model_name = 'Qwen/Qwen3-VL-235B-A22B-Instruct'
        else:  # dashscope
            api_key = self.dash_api_key
            base_url = self.dash_base_url
            if model_name is None:
                model_name = "qwen3-vl-plus"

        base64_image = self.encode_image(image_path)

        client = OpenAI(api_key=api_key, base_url=base_url)

        # 构造多模态消息体：包含图片和文本提示
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 配置额外参数（如是否开启思考过程、是否使用高清图像）
        extra_body = {
            'enable_thinking': False,
            "vl_high_resolution_images": high_resolution
        }

        # 调用API并获取响应
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            extra_body=extra_body
        )

        return response.choices[0].message.content

    def parse_json_response(self, text: str):
        """
        从可能被markdown代码块语法包围的字符串中提取纯JSON内容

        参数:
            text (str): 输入字符串，可能包含类似```json ... ```的markdown代码围栏

        返回:
            str: 清理后的纯JSON字符串，去除了周围的markdown标记
        """
        if not text:
            return ""

        # 解析markdown代码围栏结构
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                # 找到json代码块开始位置，并提取后续内容
                json_output = "\n".join(lines[i+1:])
                # 去除结束标记后的内容
                json_output = json_output.split("```")[0]
                return json_output.strip()
            elif line.strip().startswith("```") and "json" in line.strip():
                # 处理在同一行内的```json格式
                json_part = line.split("```")[1].replace("json", "").strip()
                remaining = "\n".join(lines[i+1:])
                json_output = json_part + "\n" + remaining
                json_output = json_output.split("```")[0]
                return json_output.strip()

        # 若无代码块，则尝试识别原始JSON对象或数组
        text = text.strip()
        if text.startswith('[') and text.endswith(']'):
            return text
        elif text.startswith('{') and text.endswith('}'):
            return text

        # 最终尝试通过括号匹配提取JSON片段
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx+1]

        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx+1]

        return text

    def parse_bbox_3d_from_text(self, text: str):
        """
        从响应文本中解析3D边界框信息

        参数:
            text (str): 包含3D边界框数据的响应文本

        返回:
            list: 解析得到的3D边界框列表，若解析失败则返回空列表
        """
        try:
            json_str = self.parse_json_response(text)
            if not json_str:
                return []

            # 解析JSON数据
            bbox_data = json.loads(json_str)

            # 统一输出为列表形式
            if isinstance(bbox_data, list):
                return bbox_data
            elif isinstance(bbox_data, dict):
                return [bbox_data]
            else:
                return []

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"解析3D边界框失败: {e}")
            return []

    def parse_bbox_2d_from_text(self, text: str):
        """
        从响应文本中解析2D边界框信息

        参数:
            text (str): 包含2D边界框数据的响应文本

        返回:
            list: 解析得到的2D边界框列表，若解析失败则返回空列表
        """
        try:
            json_str = self.parse_json_response(text)
            if not json_str:
                return []

            # 解析JSON数据
            bbox_data = json.loads(json_str)

            # 统一输出为列表形式
            if isinstance(bbox_data, list):
                return bbox_data
            elif isinstance(bbox_data, dict):
                return [bbox_data]
            else:
                return []

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"解析2D边界框失败: {e}")
            return []